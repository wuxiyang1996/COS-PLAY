"""
Skill retrieval indices for the versioned bank.

Maintains keyword and (optionally) embedding indices that are rebuilt
after each bank commit. Provides fast lookup used by the Decision Agent
during rollout collection.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> Set[str]:
    return {w for w in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(w) >= 2}


class KeywordIndex:
    """Inverted keyword index over skill IDs and effect literals.

    Provides fast candidate retrieval for the decode stage.
    """

    def __init__(self):
        self._id_tokens: Dict[str, Set[str]] = {}
        self._effect_tokens: Dict[str, Set[str]] = {}
        self._all_skill_ids: List[str] = []

    def build(self, bank: Any) -> None:
        """Build/rebuild the index from a bank instance."""
        self._id_tokens.clear()
        self._effect_tokens.clear()
        self._all_skill_ids = list(getattr(bank, "skill_ids", []))

        for sid in self._all_skill_ids:
            self._id_tokens[sid] = _tokenize(sid)
            contract = bank.get_contract(sid) if hasattr(bank, "get_contract") else None
            if contract is not None:
                effect_tokens: Set[str] = set()
                for eff_set in (
                    getattr(contract, "eff_add", set()) or set(),
                    getattr(contract, "eff_del", set()) or set(),
                    getattr(contract, "eff_event", set()) or set(),
                ):
                    for lit in eff_set:
                        effect_tokens |= _tokenize(lit)
                self._effect_tokens[sid] = effect_tokens
            else:
                self._effect_tokens[sid] = set()

        logger.debug("KeywordIndex: built for %d skills", len(self._all_skill_ids))

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """Search for skills matching a query string.

        Returns list of (score, skill_id) sorted by descending score.
        """
        q_tokens = _tokenize(query)
        if not q_tokens:
            return [(0.0, sid) for sid in self._all_skill_ids[:top_k]]

        scored = []
        for sid in self._all_skill_ids:
            id_tok = self._id_tokens.get(sid, set())
            eff_tok = self._effect_tokens.get(sid, set())

            id_overlap = len(q_tokens & id_tok) / max(len(q_tokens | id_tok), 1)
            eff_overlap = len(q_tokens & eff_tok) / max(len(q_tokens | eff_tok), 1)
            score = 0.6 * id_overlap + 0.4 * eff_overlap
            scored.append((score, sid))

        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    @property
    def size(self) -> int:
        return len(self._all_skill_ids)


class EmbeddingIndex:
    """Optional embedding-based index for semantic skill retrieval.

    Uses cosine similarity over skill embeddings. Requires an embedding
    function to be provided.
    """

    def __init__(self):
        self._embeddings: Dict[str, List[float]] = {}
        self._all_skill_ids: List[str] = []

    def build(
        self,
        bank: Any,
        embed_fn: Any = None,
    ) -> None:
        """Build the embedding index.

        Args:
            bank: SkillBankMVP
            embed_fn: callable(text) -> list[float]; if None, index is empty
        """
        self._embeddings.clear()
        self._all_skill_ids = list(getattr(bank, "skill_ids", []))

        if embed_fn is None:
            return

        for sid in self._all_skill_ids:
            contract = bank.get_contract(sid) if hasattr(bank, "get_contract") else None
            text = sid
            if contract:
                effects = sorted(
                    (getattr(contract, "eff_add", set()) or set())
                    | (getattr(contract, "eff_del", set()) or set())
                )
                text = f"{sid} {' '.join(effects)}"
            try:
                emb = embed_fn(text)
                self._embeddings[sid] = emb
            except Exception:
                pass

        logger.debug("EmbeddingIndex: built for %d skills", len(self._embeddings))

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[tuple]:
        """Search by cosine similarity."""
        if not self._embeddings or not query_embedding:
            return []

        import numpy as np
        q = np.array(query_embedding, dtype=np.float64)
        q_norm = np.linalg.norm(q) + 1e-9

        scored = []
        for sid, emb in self._embeddings.items():
            e = np.array(emb, dtype=np.float64)
            e_norm = np.linalg.norm(e) + 1e-9
            sim = float(np.dot(q, e) / (q_norm * e_norm))
            scored.append((sim, sid))

        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    @property
    def size(self) -> int:
        return len(self._embeddings)
