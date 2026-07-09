"""Abstract base classes for RAG and multimodal embedders."""

from abc import ABC, abstractmethod
from typing import Any, List, Union

import numpy as np


class TextEmbedderBase(ABC):
    """Base class for text-only embedding models (e.g. for RAG over experience summaries)."""

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        prompt_name: str = "passage",
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode text(s) into embedding vectors.

        Args:
            texts: Single string or list of strings to embed.
            prompt_name: Optional prompt type (e.g. "query" vs "passage" for asymmetric retrieval).
            **kwargs: Model-specific options.

        Returns:
            Array of shape (n_texts, dim).
        """
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the output embedding dimension."""
        pass


class MultimodalEmbedderBase(ABC):
    """Base class for multimodal embedding models (text + image / video)."""

    @abstractmethod
    def process(
        self,
        inputs: List[dict],
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed a list of multimodal inputs (text, image, video, or combinations).

        Each input is a dict with optional keys: "text", "image", "video", "instruction".

        Args:
            inputs: List of dicts, e.g. [{"text": "..."}, {"image": path_or_url}, {"text": "...", "image": ...}].
            normalize: Whether to L2-normalize embeddings.

        Returns:
            Array of shape (n_inputs, dim).
        """
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the output embedding dimension."""
        pass
