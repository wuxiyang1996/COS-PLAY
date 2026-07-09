"""OSWorld grounding adapter: desktop screenshot → OmniParser-v2 → schema.

Head 3 adapter for the ``desktop`` domain. Wraps
:func:`browsergym_wrapper.grounding.grounding_image_to_schema` (which
runs OmniParser-v2 locally and walks the detected
:class:`vlm_wrapper.grounding.ScreenElement` list into the canonical
``<state>…</state>`` schema) and binds ``domain="desktop"`` on the way
in. The element-to-schema bookkeeping is shared with BrowserGym because
the only thing that differs between the two domains is the schema
header.

Usage::

    from osworld_wrapper.grounding import grounding_osworld_obs_to_schema

    result = grounding_osworld_obs_to_schema(
        osworld_obs, step=1, task_id="install-spotify",
    )
    print(result["schema"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from PIL import Image

from browsergym_wrapper.grounding import grounding_image_to_schema

logger = logging.getLogger(__name__)


def grounding_osworld_obs_to_schema(
    obs: Dict[str, Any],
    *,
    step: int = 0,
    task_id: str = "",
    max_entities: int = 25,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = False,
    caption_icons: bool = True,
) -> Dict[str, Any]:
    """Adapter for OSWorld observations (from ``OSWorldGymWrapper``).

    Parameters
    ----------
    obs : dict
        OSWorld observation with ``screenshot`` (np.ndarray),
        ``instruction``, ``accessibility_tree``, ``terminal``.

    Returns
    -------
    dict — same as :func:`browsergym_wrapper.grounding.grounding_image_to_schema`.
    """
    screenshot = obs.get("screenshot")
    if screenshot is None:
        return {
            "schema": None, "elements": [],
            "warnings": ["no screenshot"], "model": "omniparser-v2",
        }

    if isinstance(screenshot, np.ndarray):
        image = Image.fromarray(screenshot)
    elif isinstance(screenshot, Image.Image):
        image = screenshot
    else:
        return {
            "schema": None, "elements": [],
            "warnings": ["unknown screenshot type"], "model": "omniparser-v2",
        }

    goal = obs.get("instruction", "")

    return grounding_image_to_schema(
        image,
        goal=goal,
        task_id=task_id,
        step=step,
        domain="desktop",
        max_entities=max_entities,
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        use_paddleocr=use_paddleocr,
        caption_icons=caption_icons,
    )


__all__ = ["grounding_osworld_obs_to_schema"]
