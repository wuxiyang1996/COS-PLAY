"""Configuration for RAG embedding models.

Defaults:
- RAG (text) embedding: Qwen/Qwen3-Embedding-0.6B
- Multimodal embedding: Qwen/Qwen3-VL-Embedding-2B

Override via environment variables or pass model names when creating embedders.
"""

import os

# Default model for text / RAG embedding (experience summaries, queries)
RAG_EMBEDDING_MODEL = os.environ.get(
    "RAG_EMBEDDING_MODEL",
    "Qwen/Qwen3-Embedding-0.6B",
)

# Default model for multimodal embedding (text + image / video)
MULTIMODAL_EMBEDDING_MODEL = os.environ.get(
    "MULTIMODAL_EMBEDDING_MODEL",
    "Qwen/Qwen3-VL-Embedding-2B",
)

# Optional: custom embedding dimension (if supported by model)
# RAG_EMBEDDING_DIM and MULTIMODAL_EMBEDDING_DIM can be set by embedders that support MRL
