"""Set-of-Marks (SoM) visual grounding for OSWorld.

Set-of-Marks is the technique behind every published OSWorld baseline that
beats the raw-pixel ceiling (~5%). Instead of asking the VLM to predict
click coordinates ``(x, y)`` from a textual schema — which it does poorly
— we:

  1. Extract every interactive element from the live AT-SPI tree (using
     :mod:`osworld_wrapper.heuristic`'s already-vetted role classifier).
  2. Draw a numbered red bounding box around each element on the
     screenshot before sending it to the VLM.
  3. Replace the action vocabulary with ``click_element(id=N)`` / etc.,
     where ``N`` is the badge number on the overlay.
  4. On execute, translate the SoM id back to ``pyautogui.click(cx, cy)``
     using the element's bbox centre.

VLMs are excellent at picking from a labelled list and poor at predicting
raw coordinates; SoM exploits that asymmetry. The OSWorld paper and the
VisualWebArena follow-up both report ~13 percentage-point gains over the
raw-pixel approach (~5% → ~18% pass@1 with the same backbone).

Public API
~~~~~~~~~~
- :func:`extract_som_elements` — AT-SPI XML → ``[SomElement]``
- :func:`draw_som_overlay` — PIL image + elements → annotated PIL image
- :func:`format_som_table` — text rendering of the element table for the
  action prompt
- :func:`som_action_to_pyautogui` — ``click_element(7)`` →
  ``pyautogui.click(cx, cy)``
- :func:`som_action_strings` — list of usable SoM action verbs to feed
  the candidate-action vocabulary

This module is intentionally side-effect-free and importable in isolation
(no OSWorld dependency); the parent actor calls it with whatever
``obs["accessibility_tree"]`` it just received.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

from .heuristic import (
    _collect_entities,
    SOM_CLICKABLE_ROLES,
)


# ── Element record ────────────────────────────────────────────────────────


@dataclass
class SomElement:
    """A single click target on the SoM overlay.

    ``som_id`` is the 1-indexed badge drawn on the screenshot; the VLM
    references it via ``click_element(id=<som_id>)``. ``bbox`` is
    ``(x, y, w, h)`` in screen-pixel coordinates as reported by
    AT-SPI's ``cp:screencoord`` / ``cp:size``.
    """

    som_id: int
    label: str
    role: str
    bbox: Tuple[int, int, int, int]

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + max(1, w // 2), y + max(1, h // 2))


# ── Extraction ────────────────────────────────────────────────────────────


# Roles where typing/text-entry is meaningful (so the prompt should
# advertise ``type_into_element`` for them).
_TYPABLE_ROLES = frozenset({
    "text", "entry", "password-text", "edit-bar", "spin-button",
    "combo-box",
})


def _is_typable(role: str) -> bool:
    return role in _TYPABLE_ROLES


def extract_som_elements(
    accessibility_xml: str,
    *,
    max_elements: int = 30,
) -> List[SomElement]:
    """Parse AT-SPI XML and return a numbered list of click targets.

    Filters to clickable roles (push-button, link, menu-item, top-level
    ``menu`` / ``menu-button`` openers, …) with a valid ``(x, y, w, h)``
    bbox. Empty / off-screen / zero-area boxes are dropped, as are
    full-screen catch-all containers (boxes covering >70% of the
    1920×1080 viewport — these correspond to the document body or the
    application frame and waste a SoM slot). The returned list is
    truncated to ``max_elements`` so the overlay stays readable.

    The role set is :data:`SOM_CLICKABLE_ROLES` from ``heuristic.py``,
    which is :data:`INTERACTIVE_ROLES` plus the classic-desktop menu
    openers. Without those, every step in the May-2026 cold-start run
    collapsed onto the GNOME dock + window decorations and the agent
    had no way to open File / Edit / Filters menus at all.
    """
    if not accessibility_xml:
        return []
    try:
        root = ET.fromstring(accessibility_xml)
    except ET.ParseError:
        return []

    raw = _collect_entities(root, max_entities=400, only_visible=True)
    out: List[SomElement] = []
    sid = 1
    seen_bboxes: set = set()
    # Filter out catch-all containers that would occupy the entire
    # screen — they are clickable in principle but the SoM badge would
    # land on top of the document body and confuse the model.
    _MAX_BBOX_FRAC = 0.70
    _SCREEN_AREA_GUESS = 1920 * 1080  # OSWorld default, conservative upper
    for e in raw:
        if e.role not in SOM_CLICKABLE_ROLES:
            continue
        if e.bbox is None:
            continue
        x, y, w, h = e.bbox
        if w <= 1 or h <= 1:
            continue
        if (w * h) >= _MAX_BBOX_FRAC * _SCREEN_AREA_GUESS:
            continue
        # Drop near-duplicate bboxes (AT-SPI sometimes reports nested wrappers)
        key = (x // 4, y // 4, w // 4, h // 4)
        if key in seen_bboxes:
            continue
        seen_bboxes.add(key)

        out.append(SomElement(
            som_id=sid,
            label=(e.label or e.role)[:60],
            role=e.role,
            bbox=(int(x), int(y), int(w), int(h)),
        ))
        sid += 1
        if sid > max_elements:
            break
    return out


# ── Overlay rendering ─────────────────────────────────────────────────────


def _resolve_font(size: int):
    if ImageFont is None:
        return None
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    )
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _measure_text(draw, text: str, font) -> Tuple[int, int]:
    """Pillow has changed the text-size API across versions. Try all."""
    if font is None:
        return (8 * len(text), 12)
    if hasattr(draw, "textbbox"):
        try:
            box = draw.textbbox((0, 0), text, font=font)
            return (box[2] - box[0], box[3] - box[1])
        except Exception:
            pass
    if hasattr(font, "getbbox"):
        try:
            box = font.getbbox(text)
            return (box[2] - box[0], box[3] - box[1])
        except Exception:
            pass
    if hasattr(draw, "textsize"):
        try:
            return draw.textsize(text, font=font)
        except Exception:
            pass
    return (8 * len(text), 12)


def draw_som_overlay(
    image,
    elements: List[SomElement],
    *,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    label_bg: Tuple[int, int, int] = (255, 255, 255),
    label_fg: Tuple[int, int, int] = (200, 0, 0),
    box_width: int = 2,
    font_size: int = 14,
):
    """Return a copy of ``image`` with numbered red boxes around each element.

    The badge ("1", "2", …) is drawn just above-left of each box on a
    white background so it's readable on any wallpaper. If the box is
    near the top edge, the badge is drawn just below-left instead so it
    doesn't get clipped off-screen.
    """
    if Image is None or ImageDraw is None:
        # Pillow unavailable; surface the screenshot unchanged.
        return image

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _resolve_font(font_size)

    W, H = img.size
    for e in elements:
        x, y, w, h = e.bbox
        x2, y2 = x + w, y + h
        # Clip rectangle so we never draw outside the image.
        rx, ry = max(0, x), max(0, y)
        rx2, ry2 = min(W - 1, x2), min(H - 1, y2)
        if rx2 <= rx or ry2 <= ry:
            continue
        draw.rectangle([(rx, ry), (rx2, ry2)], outline=box_color, width=box_width)

        tag = str(e.som_id)
        tw, th = _measure_text(draw, tag, font)
        pad = 2
        bw, bh = tw + 2 * pad, th + 2 * pad
        # Default badge: top-left, slightly above the box if room exists.
        bx, by = rx, max(0, ry - bh - 1)
        if by == 0 and ry < bh + 2:  # box is near the top — drop badge inside
            by = ry + 1
        draw.rectangle(
            [(bx, by), (bx + bw, by + bh)],
            fill=label_bg, outline=box_color, width=1,
        )
        if font is not None:
            draw.text((bx + pad, by + pad), tag, fill=label_fg, font=font)
        else:
            draw.text((bx + pad, by + pad), tag, fill=label_fg)
    return img


# ── Action vocabulary helpers ─────────────────────────────────────────────


# Regex matching the SoM action verbs we accept from the LLM (case-tolerant
# whitespace, optional underscores). Capture group 1 is always the id.
_CLICK_RE = re.compile(
    r"\bclick_?element\(\s*(?:id\s*=\s*)?(\d+)\s*\)", re.IGNORECASE,
)
_DCLICK_RE = re.compile(
    r"\b(?:double_?click|doubleclick)_?element\(\s*(?:id\s*=\s*)?(\d+)\s*\)",
    re.IGNORECASE,
)
_RCLICK_RE = re.compile(
    r"\b(?:right_?click)_?element\(\s*(?:id\s*=\s*)?(\d+)\s*\)",
    re.IGNORECASE,
)
_TYPE_RE = re.compile(
    r"\btype_?(?:text_?)?(?:into_?)?element\(\s*"
    r"(?:id\s*=\s*)?(\d+)\s*,\s*"
    r"(?:text\s*=\s*)?['\"](.*?)['\"]\s*\)",
    re.IGNORECASE,
)


def som_action_strings(elements: List[SomElement]) -> List[str]:
    """Return the SoM action verbs to add to the candidate-action list.

    We expose at most 12 click variants (one per top SoM element) plus a
    generic ``type_into_element(id=N, text='...')`` template, so the
    candidate list stays under 22 entries (the existing cap).
    """
    out: List[str] = []
    for e in elements[:12]:
        out.append(f"click_element(id={e.som_id})  # {e.role}: {e.label[:40]}")
    if any(_is_typable(e.role) for e in elements):
        out.append("type_into_element(id=N, text='...')")
    return out


def som_action_to_pyautogui(
    action_string: str,
    elements: List[SomElement],
) -> Optional[str]:
    """Translate a SoM action verb to a concrete ``pyautogui`` call.

    Returns ``None`` if ``action_string`` is not a recognised SoM verb,
    so the caller can fall back to executing the raw string. If the
    referenced ``id`` doesn't exist in ``elements`` we also return
    ``None`` (caller treats it as an action-selection error).
    """
    if not action_string:
        return None

    by_id = {e.som_id: e for e in elements}

    m = _TYPE_RE.search(action_string)
    if m:
        sid = int(m.group(1))
        text = m.group(2)
        el = by_id.get(sid)
        if el is None:
            return None
        cx, cy = el.center
        # Click first (focus), then type. We escape only the obvious
        # quote chars; if the model needs literal newlines it can issue
        # a follow-up press('enter').
        safe = text.replace("\\", "\\\\").replace("'", "\\'")
        return (
            f"pyautogui.click({cx}, {cy}); "
            f"pyautogui.typewrite('{safe}', interval=0.05)"
        )

    m = _DCLICK_RE.search(action_string)
    if m:
        sid = int(m.group(1))
        el = by_id.get(sid)
        if el is None:
            return None
        cx, cy = el.center
        return f"pyautogui.doubleClick({cx}, {cy})"

    m = _RCLICK_RE.search(action_string)
    if m:
        sid = int(m.group(1))
        el = by_id.get(sid)
        if el is None:
            return None
        cx, cy = el.center
        return f"pyautogui.rightClick({cx}, {cy})"

    m = _CLICK_RE.search(action_string)
    if m:
        sid = int(m.group(1))
        el = by_id.get(sid)
        if el is None:
            return None
        cx, cy = el.center
        return f"pyautogui.click({cx}, {cy})"

    return None


def format_som_table(elements: List[SomElement]) -> str:
    """Render the element table for inclusion in the action prompt."""
    if not elements:
        return "(no SoM elements detected this step)"
    lines = [
        "Set-of-Marks elements (each numbered red box on the screenshot):",
    ]
    for e in elements:
        x, y, w, h = e.bbox
        cx, cy = e.center
        typable = " [typable]" if _is_typable(e.role) else ""
        lines.append(
            f"  [{e.som_id:>2d}] {e.role:<14s} \"{e.label[:42]}\"  "
            f"center=({cx},{cy}){typable}"
        )
    return "\n".join(lines)


__all__ = [
    "SomElement",
    "extract_som_elements",
    "draw_som_overlay",
    "format_som_table",
    "som_action_strings",
    "som_action_to_pyautogui",
]
