"""Ranking and retrieval: compare query embedding to stored memory embeddings, return top-k."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from rag.embedding.base import TextEmbedderBase, MultimodalEmbedderBase


# Default top-k when not specified
DEFAULT_TOP_K = 5


def _cosine_similarity(query_emb: np.ndarray, memory_embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity (dot product when vectors are L2-normalized)."""
    query_emb = np.atleast_2d(query_emb).astype(np.float32)
    memory_embeddings = np.asarray(memory_embeddings, dtype=np.float32)
    if query_emb.shape[0] != 1:
        raise ValueError("query_emb must be a single vector or (1, dim).")
    # (1, dim) @ (dim, n) -> (1, n)
    scores = query_emb @ memory_embeddings.T
    return np.squeeze(scores, axis=0)


def rank_memories(
    query_embedding: np.ndarray,
    memory_embeddings: np.ndarray,
    payloads: Optional[List[Any]] = None,
    k: int = DEFAULT_TOP_K,
) -> List[Tuple[int, float, Any]]:
    """Rank stored memories by similarity to the query embedding; return top-k.

    Args:
        query_embedding: Shape (dim,) or (1, dim).
        memory_embeddings: Shape (n_memories, dim).
        payloads: Optional list of n_memories objects (e.g. experience dicts). If None, indices only.
        k: Number of top results to return (hyperparameter).

    Returns:
        List of (index, score, payload) for top-k, ordered by score descending.
        payload is payloads[i] if payloads is not None else None.
    """
    k = min(k, len(memory_embeddings))
    if k <= 0:
        return []
    scores = _cosine_similarity(query_embedding, memory_embeddings)
    n = len(scores)
    if payloads is not None and len(payloads) != n:
        raise ValueError("payloads length must match memory_embeddings first dimension.")
    # top-k indices (largest first)
    if k >= n:
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
    result = []
    for idx in top_indices:
        i = int(idx)
        sc = float(scores[i])
        payload = payloads[i] if payloads is not None else None
        result.append((i, sc, payload))
    return result


class MemoryStore:
    """Store memory embeddings and rank/retrieve top-k by query similarity.

    Works with either a text embedder (for RAG over text memories) or a
    multimodal embedder (for text + image / video memories). k is a
    hyperparameter (default 5).
    """

    def __init__(
        self,
        embedder: Union[TextEmbedderBase, MultimodalEmbedderBase],
        top_k: int = DEFAULT_TOP_K,
    ):
        """
        Args:
            embedder: TextEmbedder or MultimodalEmbedder used to embed queries and (on add) contents.
            top_k: Default number of top similar memories to return (hyperparameter).
        """
        self._embedder = embedder
        self._top_k = top_k
        self._embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._payloads: List[Any] = []

    @property
    def top_k(self) -> int:
        return self._top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        self._top_k = max(0, int(value))

    def __len__(self) -> int:
        return len(self._payloads)

    def _embed_query_text(self, query: str) -> np.ndarray:
        if isinstance(self._embedder, TextEmbedderBase):
            return self._embedder.encode(query, prompt_name="query")
        raise TypeError("Query is text but embedder is multimodal; use a dict with 'text' key or TextEmbedder.")

    def _embed_query_multimodal(self, query: Union[dict, str]) -> np.ndarray:
        if isinstance(self._embedder, MultimodalEmbedderBase):
            inp = query if isinstance(query, dict) else {"text": query}
            return self._embedder.process([inp], normalize=True)
        if isinstance(self._embedder, TextEmbedderBase):
            text = query.get("text", "") if isinstance(query, dict) else query
            return self._embedder.encode(text, prompt_name="query")
        raise TypeError("Unsupported embedder type.")

    def add_texts(
        self,
        texts: List[str],
        payloads: Optional[List[Any]] = None,
    ) -> None:
        """Add text memories (for use with TextEmbedder). Embed and store."""
        if not isinstance(self._embedder, TextEmbedderBase):
            raise TypeError("add_texts requires a TextEmbedder.")
        embeddings = self._embedder.encode(texts, prompt_name="passage")
        self.add_embeddings(embeddings, payloads if payloads is not None else texts)

    def add_multimodal(
        self,
        inputs: List[dict],
        payloads: Optional[List[Any]] = None,
    ) -> None:
        """Add multimodal memories (for use with MultimodalEmbedder). Embed and store."""
        if not isinstance(self._embedder, MultimodalEmbedderBase):
            raise TypeError("add_multimodal requires a MultimodalEmbedder.")
        embeddings = self._embedder.process(inputs, normalize=True)
        self.add_embeddings(embeddings, payloads if payloads is not None else inputs)

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        payloads: Optional[List[Any]] = None,
    ) -> None:
        """Append precomputed embeddings and optional payloads."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D (n, dim).")
        n = embeddings.shape[0]
        if payloads is not None and len(payloads) != n:
            raise ValueError("payloads length must match embeddings first dimension.")
        if len(self._payloads) == 0:
            self._embeddings = embeddings
            self._payloads = list(payloads) if payloads is not None else [None] * n
        else:
            dim = self._embeddings.shape[1]
            if embeddings.shape[1] != dim:
                raise ValueError(f"New embeddings dim {embeddings.shape[1]} must match store dim {dim}.")
            self._embeddings = np.vstack([self._embeddings, embeddings])
            self._payloads.extend(payloads if payloads is not None else [None] * n)

    def retrieve(
        self,
        query: Union[str, dict],
        k: Optional[int] = None,
        return_scores: bool = True,
    ) -> List[Any]:
        """Return top-k most similar memories to the query.

        Args:
            query: For text embedder: a string. For multimodal: a dict with optional
                   "text", "image", "video", "instruction".
            k: Number of results (default: store's top_k hyperparameter).
            return_scores: If True, return list of (payload, score); else list of payloads.

        Returns:
            List of (payload, score) or list of payloads for top-k memories.
        """
        k = k if k is not None else self._top_k
        k = min(k, len(self._payloads))
        if k <= 0:
            return [] if not return_scores else []

        if isinstance(self._embedder, TextEmbedderBase) and isinstance(query, str):
            query_emb = self._embedder.encode(query, prompt_name="query")
        else:
            query_emb = self._embed_query_multimodal(query)
        if query_emb.ndim == 2:
            query_emb = np.squeeze(query_emb, axis=0)

        ranked = rank_memories(
            query_emb,
            self._embeddings,
            payloads=self._payloads,
            k=k,
        )
        if return_scores:
            return [(payload, score) for _, score, payload in ranked]
        return [payload for _, _, payload in ranked]

    def rank(
        self,
        query: Union[str, dict],
        k: Optional[int] = None,
    ) -> List[Tuple[int, float, Any]]:
        """Return top-k (index, score, payload) for the query. k is the hyperparameter."""
        k = k if k is not None else self._top_k
        k = min(k, len(self._payloads))
        if k <= 0:
            return []

        if isinstance(self._embedder, TextEmbedderBase) and isinstance(query, str):
            query_emb = self._embedder.encode(query, prompt_name="query")
        else:
            query_emb = self._embed_query_multimodal(query)
        if query_emb.ndim == 2:
            query_emb = np.squeeze(query_emb, axis=0)

        return rank_memories(
            query_emb,
            self._embeddings,
            payloads=self._payloads,
            k=k,
        )


def get_memory_store(
    embedder: Union[TextEmbedderBase, MultimodalEmbedderBase],
    top_k: int = DEFAULT_TOP_K,
) -> MemoryStore:
    """Factory: build a MemoryStore with the given embedder and top_k hyperparameter."""
    return MemoryStore(embedder=embedder, top_k=top_k)
