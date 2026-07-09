# RAG module: embedding models and experience retrieval for Game-AI-Agent.
# - Text (RAG) embedding: default Qwen/Qwen3-Embedding-0.6B
# - Multimodal embedding: default Qwen/Qwen3-VL-Embedding-2B
# - Ranking: compare query to stored memory embeddings, return top-k

from rag.embedding.base import TextEmbedderBase, MultimodalEmbedderBase
from rag.embedding.text_embedder import TextEmbedder, get_text_embedder
from rag.embedding.multimodal_embedder import MultimodalEmbedder, get_multimodal_embedder
from rag.config import RAG_EMBEDDING_MODEL, MULTIMODAL_EMBEDDING_MODEL
from rag.retrieval import (
    MemoryStore,
    rank_memories,
    get_memory_store,
    DEFAULT_TOP_K,
)

__all__ = [
    "TextEmbedderBase",
    "MultimodalEmbedderBase",
    "TextEmbedder",
    "MultimodalEmbedder",
    "get_text_embedder",
    "get_multimodal_embedder",
    "RAG_EMBEDDING_MODEL",
    "MULTIMODAL_EMBEDDING_MODEL",
    "MemoryStore",
    "rank_memories",
    "get_memory_store",
    "DEFAULT_TOP_K",
]
