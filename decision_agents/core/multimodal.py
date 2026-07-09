"""Multimodal input scaffolding shared by SFT / GRPO actors.

A :class:`VisualInput` is the per-step image attachment the actor sees
in addition to the parsed ``<state>`` schema.  Both flavours use the
same dataclass so SFT records and GRPO rollouts stay 1:1 alignable —
critical for the distillation pipeline described in
``vlm_wrapper/README.md`` (Stage A: schema-only SFT; Stage B:
tool-trajectory SFT).

Two content-part builders are provided:

* :func:`build_openai_vision_messages` — OpenAI / GPT-4o chat content
  parts (``{"type": "image_url", "image_url": {"url": "data:..."}}``).
  Used by :class:`decision_agents.SFT.actor_gpt4o.GPT4oCollectorActor`.
* :func:`build_qwen_vl_messages` — Qwen3-VL chat content parts (same
  OpenAI-compatible ``image_url`` shape that vLLM accepts when serving
  ``Qwen/Qwen3-VL-8B-Instruct``).  Used by
  :class:`decision_agents.grpo.actor_qwen_vl.QwenVLActor`.

Both builders return the *messages* list rather than just the content
parts so callers can drop the result straight into ``chat.completions``
without re-wrapping.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

_LOGGER = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


# ──────────────────────────────────────────────────────────────────────
# VisualInput
# ──────────────────────────────────────────────────────────────────────


@dataclass
class VisualInput:
    """One screenshot / frame attached to an actor step.

    At least one of ``image_path``, ``image_b64``, or ``image_url`` must
    be set.  Helpers in this module accept any of them and normalise to
    the chat-completion content part shape both GPT-4o and vLLM /
    Qwen3-VL accept (``{"type": "image_url", "image_url": {"url": ...}}``).

    Fields
    ------
    image_path
        Local filesystem path.  Preferred when the image lives next to
        the rollout artefact (vLLM and GPT-4o both inline it as base64
        data URL when serialised).
    image_b64
        Already-base64-encoded image bytes (no ``data:...`` prefix).
        Used by callers that hold the screenshot in memory.
    image_url
        Remote URL.  Only valid when the OpenAI / vLLM endpoint can
        reach it; for offline rollouts prefer ``image_path``.
    mime_type
        Optional override for the data URL prefix (``image/png`` /
        ``image/jpeg``).  Auto-detected from ``image_path`` when omitted.
    width / height
        Optional metadata, logged on rollout records so we can detect
        resolution drift between SFT and GRPO data later.
    caption
        Optional human-readable caption.  When set, the multimodal
        builders prepend it to the text content part so the LLM has a
        textual hook for the image.  Useful for tool-result frames
        like ``"OmniParser overlay: 13 actionable elements"``.
    """

    image_path: Optional[PathLike] = None
    image_b64: Optional[str] = None
    image_url: Optional[str] = None
    mime_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    caption: Optional[str] = None

    def __post_init__(self) -> None:
        if self.image_path is None and self.image_b64 is None and self.image_url is None:
            raise ValueError(
                "VisualInput needs at least one of "
                "image_path / image_b64 / image_url"
            )

    def as_data_url(self) -> str:
        """Return a ``data:<mime>;base64,<payload>`` URL.

        Uses ``image_url`` verbatim when it's already a data URL or a
        reachable http(s) URL; otherwise loads ``image_path`` or wraps
        ``image_b64`` with the right MIME prefix.
        """
        if self.image_url:
            # Either an http(s) URL or a pre-built data URL — both are
            # accepted by OpenAI and vLLM as-is.
            return self.image_url

        mime = self.mime_type or _guess_mime(self.image_path)
        if self.image_b64:
            return f"data:{mime};base64,{self.image_b64}"
        if self.image_path:
            return load_image_as_data_url(self.image_path, mime_type=mime)
        # Unreachable thanks to __post_init__, but keep mypy happy.
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Compact log/serialisation form (skips heavy base64)."""
        return {
            "image_path": str(self.image_path) if self.image_path else None,
            "image_url": self.image_url,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "caption": self.caption,
            "has_b64": bool(self.image_b64),
        }


# ──────────────────────────────────────────────────────────────────────
# Image loading
# ──────────────────────────────────────────────────────────────────────


def _guess_mime(path: Optional[PathLike]) -> str:
    if not path:
        return "image/png"
    guess, _ = mimetypes.guess_type(str(path))
    return guess or "image/png"


def load_image_as_data_url(path: PathLike, *, mime_type: Optional[str] = None) -> str:
    """Read *path* and return a base64 ``data:`` URL.

    Wrapped in a small helper so the SFT recorder can reuse it for the
    optional ``image_b64`` field on persisted JSONL rows (lets the GRPO
    side reload images without going through the filesystem).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"VisualInput image not found: {p}")
    mime = mime_type or _guess_mime(p)
    payload = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


# ──────────────────────────────────────────────────────────────────────
# Chat-message builders
# ──────────────────────────────────────────────────────────────────────


def _content_parts(text: str, images: Sequence[VisualInput]) -> List[Dict[str, Any]]:
    """Build the OpenAI-style ``content`` array (text + images).

    Any image with a ``caption`` gets an extra small text part inserted
    before it so the LLM has a textual anchor (e.g. ``"OmniParser
    overlay:"``); this matches how the cascade in ``vlm_wrapper`` chains
    tool results.
    """
    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        if img.caption:
            parts.append({"type": "text", "text": img.caption})
        parts.append({"type": "image_url", "image_url": {"url": img.as_data_url()}})
    return parts


def build_openai_vision_messages(
    *,
    prompt: str,
    images: Optional[Sequence[VisualInput]] = None,
    system: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build a ``messages=`` list for ``openai.chat.completions.create``.

    GPT-4o accepts the standard OpenAI chat content-part shape, so this
    is just thin wrapping around :func:`_content_parts`.
    """
    msgs: List[Dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    if not images:
        msgs.append({"role": "user", "content": prompt})
    else:
        msgs.append({"role": "user", "content": _content_parts(prompt, images)})
    return msgs


def build_qwen_vl_messages(
    *,
    prompt: str,
    images: Optional[Sequence[VisualInput]] = None,
    system: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build a ``messages=`` list compatible with Qwen3-VL on vLLM.

    vLLM's OpenAI-compatible endpoint serving ``Qwen/Qwen3-VL-8B-Instruct``
    accepts the same ``[{"type": "image_url", "image_url": {"url": ...}}]``
    content-part shape, so the builder is essentially identical to the
    OpenAI variant.  Kept as a separate function so future
    Qwen-specific tweaks (e.g. ``min_pixels`` / ``max_pixels`` hints
    that vLLM forwards as ``image_url`` extras) only touch one place.
    """
    return build_openai_vision_messages(prompt=prompt, images=images, system=system)
