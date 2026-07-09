"""Multimodal embedding (text + image / video) using Qwen3-VL-Embedding by default."""

from typing import Any, List, Optional

import numpy as np

from rag.embedding.base import MultimodalEmbedderBase
from rag.config import MULTIMODAL_EMBEDDING_MODEL
from rag.embedding.qwen3_vl_embedding import (
    Qwen3VLEmbedder,
    is_qwen3_vl_available,
    get_qwen3_vl_import_error,
)


class MultimodalEmbedder(MultimodalEmbedderBase):
    """Multimodal embedder for text, images, and video.

    Default model: Qwen/Qwen3-VL-Embedding-2B. Each input is a dict with optional
    keys: "text", "image", "video", "instruction".
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Args:
            model_name_or_path: Hugging Face model id or path. Defaults to MULTIMODAL_EMBEDDING_MODEL.
            **kwargs: Passed to the underlying embedder (e.g. max_length, default_instruction).
        """
        if not is_qwen3_vl_available():
            err = get_qwen3_vl_import_error()
            raise ImportError(
                "Multimodal embedding requires transformers (with Qwen3-VL), qwen-vl-utils, and torch. "
                "Install with: pip install 'transformers>=4.57' qwen-vl-utils torch"
            ) from err
        self._model_name = model_name_or_path or MULTIMODAL_EMBEDDING_MODEL
        self._kwargs = kwargs
        self._embedder: Optional[Qwen3VLEmbedder] = None

    def _get_embedder(self) -> Qwen3VLEmbedder:
        if self._embedder is None:
            self._embedder = Qwen3VLEmbedder(self._model_name, **self._kwargs)
        return self._embedder

    def process(
        self,
        inputs: List[dict],
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed a list of multimodal inputs.

        Each item in inputs is a dict with optional keys:
          - "text": str
          - "image": str (path/url) or PIL Image
          - "video": str (path/url) or list of frames
          - "instruction": str (optional task instruction)
        """
        embedder = self._get_embedder()
        out = embedder.process(inputs, normalize=normalize)
        if hasattr(out, "cpu"):
            out = out.cpu()
        if hasattr(out, "numpy"):
            out = out.numpy()
        return np.asarray(out, dtype=np.float32)

    @property
    def embedding_dimension(self) -> int:
        embedder = self._get_embedder()
        cfg = embedder.model.config
        return getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", 2048)


def get_multimodal_embedder(
    model_name_or_path: Optional[str] = None,
    **kwargs: Any,
) -> MultimodalEmbedder:
    """Factory: return a MultimodalEmbedder with optional overrides.

    Uses MULTIMODAL_EMBEDDING_MODEL (or env MULTIMODAL_EMBEDDING_MODEL) when model_name_or_path is None.
    """
    return MultimodalEmbedder(
        model_name_or_path=model_name_or_path or MULTIMODAL_EMBEDDING_MODEL,
        **kwargs,
    )
