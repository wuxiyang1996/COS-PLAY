"""Text (RAG) embedding using Qwen3-Embedding by default, with support for other models."""

import os
from typing import Any, List, Optional, Union

import numpy as np

from rag.embedding.base import TextEmbedderBase
from rag.config import RAG_EMBEDDING_MODEL


class TextEmbedder(TextEmbedderBase):
    """Text embedder for RAG: experience summaries, queries, etc.

    Default model: Qwen/Qwen3-Embedding-0.6B. Use any SentenceTransformer-compatible
    or Hugging Face text embedding model by passing model_name_or_path.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        use_flash_attention: bool = False,
        **model_kwargs: Any,
    ):
        """
        Args:
            model_name_or_path: Hugging Face model id or path. Defaults to RAG_EMBEDDING_MODEL.
            device: Device to run on (e.g. "cuda", "cpu"). ``None`` checks the
                ``RAG_EMBEDDER_DEVICE`` env var, then falls back to SentenceTransformer auto.
            use_flash_attention: If True, enable flash_attention_2 when supported.
            **model_kwargs: Passed to SentenceTransformer (e.g. trust_remote_code=True).
        """
        self._model_name = model_name_or_path or RAG_EMBEDDING_MODEL
        self._device = device or os.environ.get("RAG_EMBEDDER_DEVICE") or None
        self._use_flash_attention = use_flash_attention
        self._model_kwargs = model_kwargs
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Text embedding requires sentence-transformers. Install with: pip install sentence-transformers"
            ) from e

        kwargs = dict(self._model_kwargs)
        if self._use_flash_attention:
            kwargs.setdefault("model_kwargs", {})["attn_implementation"] = "flash_attention_2"
            kwargs.setdefault("model_kwargs", {})["device_map"] = "auto"
            kwargs.setdefault("tokenizer_kwargs", {})["padding_side"] = "left"

        self._model = SentenceTransformer(self._model_name, device=self._device, **kwargs)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False
        return self._model

    def encode(
        self,
        texts: Union[str, List[str]],
        prompt_name: str = "passage",
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode text(s) into normalized embedding vectors.

        For retrieval, use prompt_name="query" for query strings and "passage" (default)
        for documents.
        """
        model = self._get_model()
        if isinstance(texts, str):
            texts = [texts]
        # SentenceTransformer.encode returns numpy by default; prompt_name used by Qwen3 etc.
        encode_kw = dict(kwargs)
        if getattr(model, "prompts", None) and prompt_name in (model.prompts or {}):
            encode_kw["prompt_name"] = prompt_name
        emb = model.encode(texts, **encode_kw)
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        return np.asarray(emb, dtype=np.float32)

    @property
    def embedding_dimension(self) -> int:
        model = self._get_model()
        return model.get_sentence_embedding_dimension()


_shared_cpu_embedder: Optional[TextEmbedder] = None


def get_text_embedder(
    model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
    shared: bool = False,
    **kwargs: Any,
) -> TextEmbedder:
    """Factory: return a TextEmbedder with optional overrides.

    Uses RAG_EMBEDDING_MODEL (or env RAG_EMBEDDING_MODEL) when model_name_or_path is None.

    Parameters
    ----------
    device : str, optional
        Explicitly set the device (e.g. ``"cpu"``).  When *None*,
        ``SentenceTransformer`` picks the default (usually ``cuda:0``).
    shared : bool
        If True and device is ``"cpu"``, return a module-level singleton
        so multiple callers share one model instance.
    """
    global _shared_cpu_embedder
    resolved_model = model_name_or_path or RAG_EMBEDDING_MODEL
    if shared and device == "cpu":
        if _shared_cpu_embedder is None:
            _shared_cpu_embedder = TextEmbedder(
                model_name_or_path=resolved_model, device="cpu", **kwargs,
            )
        return _shared_cpu_embedder
    return TextEmbedder(model_name_or_path=resolved_model, device=device, **kwargs)
