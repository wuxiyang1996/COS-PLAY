"""OmniParser-v2 screen grounding: screenshot → structured UI elements.

Head 3 of the vlm_wrapper pipeline.  Uses Microsoft's OmniParser-v2
(YOLO icon detector + Florence-2 icon captioner + OCR) to parse a GUI
screenshot into a list of ScreenElement objects with bounding boxes,
semantic labels, and interactivity flags — all locally, no API calls.

Model weights are downloaded from HuggingFace on first use and cached
under ``~/.cache/omniparser-v2/``.

Usage::

    from vlm_wrapper.grounding import parse_screen, ScreenElement

    elements = parse_screen(pil_image)
    for el in elements:
        print(el.label, el.bbox, el.element_type, el.interactable)

Requirements:
    pip install ultralytics transformers easyocr torch torchvision
    pip install supervision   # for box annotation / overlap removal
    pip install paddleocr     # optional, for PaddleOCR backend

Hardware:
    - GPU (>=6 GB VRAM): ~0.6 s/frame  (A100: 0.4 s)
    - CPU-only: ~4-8 s/frame (usable for offline labeling)
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────

DEFAULT_CACHE_DIR = os.environ.get(
    "OMNIPARSER_CACHE_DIR",
    str(Path.home() / ".cache" / "omniparser-v2"),
)
HF_REPO_ID = "microsoft/OmniParser-v2.0"

DEFAULT_BOX_THRESHOLD = 0.05
DEFAULT_IOU_THRESHOLD = 0.1
DEFAULT_IMGSZ = 640
DEFAULT_TEXT_THRESHOLD = 0.9
DEFAULT_MAX_ELEMENTS = 50
DEFAULT_CAPTION_BATCH = 64


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x2, self.y2

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def __repr__(self) -> str:
        return f"BBox(x={self.x}, y={self.y}, w={self.w}, h={self.h})"


@dataclass
class ScreenElement:
    """A single detected UI element on screen."""
    element_type: str           # "icon" or "text"
    label: str                  # semantic caption or OCR content
    bbox: BBox                  # pixel-coordinate bounding box
    interactable: bool = True   # icons are interactable, text blocks usually not
    confidence: float = 0.0
    source: str = ""            # provenance: "yolo", "ocr", "yolo+ocr"

    @property
    def bbox_ratio(self) -> None:
        """Not stored — compute from image dims if needed."""
        return None

    def __repr__(self) -> str:
        return (
            f"ScreenElement({self.element_type!r}, {self.label!r}, "
            f"{self.bbox}, inter={self.interactable})"
        )


# ── Global model cache (lazy-loaded) ─────────────────────────────────

_yolo_model = None
_caption_model_processor = None
_ocr_reader = None
_paddle_ocr = None
_device = None


def _get_device():
    global _device
    if _device is not None:
        return _device
    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("OmniParser device: %s", _device)
    return _device


def _ensure_weights(cache_dir: str = DEFAULT_CACHE_DIR) -> Path:
    """Download OmniParser-v2 weights from HuggingFace if not cached."""
    cache = Path(cache_dir)
    marker = cache / "icon_detect" / "model.pt"
    if marker.exists():
        return cache

    logger.info("Downloading OmniParser-v2 weights from %s → %s", HF_REPO_ID, cache)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_REPO_ID, local_dir=str(cache))
    if not marker.exists():
        raise RuntimeError(
            f"Weight download succeeded but {marker} not found. "
            f"Check {cache} for the expected layout."
        )
    return cache


def _load_yolo(cache_dir: str = DEFAULT_CACHE_DIR):
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    from ultralytics import YOLO
    weights = _ensure_weights(cache_dir)
    model_path = str(weights / "icon_detect" / "model.pt")
    _yolo_model = YOLO(model_path)
    logger.info("YOLO icon detector loaded from %s", model_path)
    return _yolo_model


def _load_caption(cache_dir: str = DEFAULT_CACHE_DIR):
    global _caption_model_processor
    if _caption_model_processor is not None:
        return _caption_model_processor

    # Florence-2's image backbone (DaViT) needs ``timm``.  If it's missing
    # we raise a single, actionable error here instead of letting the
    # transformers loader die deep inside ``modeling_florence2.py``.
    try:
        import timm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Florence-2 captioning requires `timm`.  Install the optional "
            "vision extra: `pip install -e .[vision]` (or `pip install timm`)."
        ) from e

    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    weights = _ensure_weights(cache_dir)
    model_path = str(weights / "icon_caption")
    device = _get_device()

    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True,
    )
    dtype = torch.float16 if device == "cuda" else torch.float32

    # transformers ≥4.50 runs `_check_and_adjust_attn_implementation`
    # during `from_pretrained`, which reads `model._supports_sdpa`.
    # Florence-2's custom `Florence2ForConditionalGeneration` (loaded
    # via trust_remote_code from microsoft/Florence-2-*) predates that
    # attribute, so the load itself crashes with `AttributeError`.
    # Forcing `attn_implementation="eager"` bypasses the SDPA path so
    # the missing attribute never gets read, and is a no-op on models
    # that already default to eager attention.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
    except (TypeError, ValueError):
        # Very old transformers that don't know `attn_implementation=`.
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)

    # Belt-and-suspenders: patch the missing attribute on the class so
    # downstream calls that re-check it (e.g. `.generate()` internals)
    # also succeed.
    for attr, default in (
        ("_supports_sdpa", False),
        ("_supports_flash_attn_2", False),
        ("_supports_cache_class", False),
    ):
        if not hasattr(type(model), attr):
            setattr(type(model), attr, default)

    _caption_model_processor = {"model": model, "processor": processor}
    logger.info("Florence-2 icon captioner loaded from %s (device=%s)", model_path, device)
    return _caption_model_processor


def _load_ocr(use_paddleocr: bool = False):
    global _ocr_reader, _paddle_ocr
    if use_paddleocr:
        if _paddle_ocr is not None:
            return "paddle", _paddle_ocr
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(
            lang="en", use_angle_cls=False, use_gpu=False,
            show_log=False, max_batch_size=1024, use_dilation=True,
            det_db_score_mode="slow", rec_batch_num=1024,
        )
        logger.info("PaddleOCR loaded")
        return "paddle", _paddle_ocr
    else:
        if _ocr_reader is not None:
            return "easyocr", _ocr_reader
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=(_get_device() == "cuda"))
        logger.info("EasyOCR loaded")
        return "easyocr", _ocr_reader


# ── Low-level detection functions ─────────────────────────────────────

def _run_yolo(
    image: Image.Image,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    imgsz: int = DEFAULT_IMGSZ,
) -> List[dict]:
    """Run YOLO icon detector.  Returns list of {bbox_ratio, confidence}."""
    import torch

    model = _load_yolo()
    w, h = image.size

    result = model.predict(
        source=image, conf=box_threshold, imgsz=imgsz, iou=0.1, verbose=False,
    )
    boxes_xyxy = result[0].boxes.xyxy   # pixel coords
    confs = result[0].boxes.conf

    detections = []
    for box, conf in zip(boxes_xyxy, confs):
        x1, y1, x2, y2 = box.tolist()
        bw, bh = x2 - x1, y2 - y1
        if bw * bh < 4:
            continue
        detections.append({
            "bbox_ratio": [x1 / w, y1 / h, x2 / w, y2 / h],
            "bbox_pixel": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": float(conf),
            "type": "icon",
        })
    return detections


def _run_ocr(
    image: Image.Image,
    use_paddleocr: bool = False,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> Tuple[List[str], List[List[int]]]:
    """Run OCR on the image.  Returns (texts, bboxes_xyxy_pixel)."""
    img_np = np.array(image.convert("RGB"))
    w, h = image.size
    backend, engine = _load_ocr(use_paddleocr)

    if backend == "paddle":
        result = engine.ocr(img_np, cls=False)[0]
        if result is None:
            return [], []
        texts = [item[1][0] for item in result if item[1][1] > text_threshold]
        coords = [item[0] for item in result if item[1][1] > text_threshold]
        bboxes = []
        for coord in coords:
            x1, y1 = int(coord[0][0]), int(coord[0][1])
            x2, y2 = int(coord[2][0]), int(coord[2][1])
            bboxes.append([x1, y1, x2, y2])
    else:
        result = engine.readtext(
            img_np, paragraph=False, text_threshold=text_threshold,
        )
        texts = [item[1] for item in result]
        bboxes = []
        for item in result:
            coord = item[0]
            x1, y1 = int(coord[0][0]), int(coord[0][1])
            x2, y2 = int(coord[2][0]), int(coord[2][1])
            bboxes.append([x1, y1, x2, y2])

    return texts, bboxes


def _caption_icons(
    image: Image.Image,
    icon_boxes_ratio: List[List[float]],
    batch_size: int = DEFAULT_CAPTION_BATCH,
) -> List[str]:
    """Caption cropped icon regions using Florence-2."""
    import torch
    from torchvision.transforms import ToPILImage
    import cv2

    if not icon_boxes_ratio:
        return []

    cap = _load_caption()
    model, processor = cap["model"], cap["processor"]
    device = model.device
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]
    to_pil = ToPILImage()

    crops = []
    for box in icon_boxes_ratio:
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        crop = img_np[max(0, y1):max(1, y2), max(0, x1):max(1, x2), :]
        if crop.size == 0:
            crop = img_np[:1, :1, :]
        crop = cv2.resize(crop, (64, 64))
        crops.append(to_pil(crop))

    prompt = " "
    captions = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        if device.type == "cuda":
            inputs = processor(
                images=batch, text=[prompt] * len(batch),
                return_tensors="pt", do_resize=False,
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(
                images=batch, text=[prompt] * len(batch),
                return_tensors="pt",
            ).to(device=device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=20, num_beams=1, do_sample=False,
            use_cache=False,  # Florence-2 + transformers ≥4.55 cache shim bug
        )
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions.extend([t.strip() for t in texts])

    return captions


# ── Overlap removal ───────────────────────────────────────────────────

def _iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two xyxy boxes (ratio or pixel)."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(1e-8, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    a2 = max(1e-8, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = a1 + a2 - inter + 1e-8
    return max(inter / union, inter / a1, inter / a2)


def _merge_detections(
    yolo_dets: List[dict],
    ocr_texts: List[str],
    ocr_bboxes: List[List[int]],
    image_size: Tuple[int, int],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> List[dict]:
    """Merge YOLO icon detections with OCR text detections, removing overlaps.

    Returns a unified list of element dicts:
        {type, label, bbox_pixel, interactable, confidence, source}
    """
    w, h = image_size

    ocr_elements = []
    for txt, box in zip(ocr_texts, ocr_bboxes):
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw * bh < 4:
            continue
        ocr_elements.append({
            "type": "text",
            "label": txt,
            "bbox_pixel": box,
            "bbox_ratio": [box[0] / w, box[1] / h, box[2] / w, box[3] / h],
            "interactable": False,
            "confidence": 1.0,
            "source": "ocr",
        })

    merged = list(ocr_elements)

    for det in yolo_dets:
        icon_ratio = det["bbox_ratio"]
        absorbed_ocr_labels = []
        skip = False
        remove_indices = []

        for idx, ocr_el in enumerate(merged):
            if ocr_el["source"] != "ocr":
                continue
            overlap = _iou(icon_ratio, ocr_el["bbox_ratio"])
            ocr_area = (ocr_el["bbox_ratio"][2] - ocr_el["bbox_ratio"][0]) * \
                       (ocr_el["bbox_ratio"][3] - ocr_el["bbox_ratio"][1])
            icon_area = (icon_ratio[2] - icon_ratio[0]) * (icon_ratio[3] - icon_ratio[1])

            # OCR box inside icon → absorb the OCR label into this icon
            inter_x1 = max(icon_ratio[0], ocr_el["bbox_ratio"][0])
            inter_y1 = max(icon_ratio[1], ocr_el["bbox_ratio"][1])
            inter_x2 = min(icon_ratio[2], ocr_el["bbox_ratio"][2])
            inter_y2 = min(icon_ratio[3], ocr_el["bbox_ratio"][3])
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            if ocr_area > 0 and inter_area / ocr_area > 0.8:
                absorbed_ocr_labels.append(ocr_el["label"])
                remove_indices.append(idx)
            elif icon_area > 0 and inter_area / icon_area > 0.8:
                skip = True
                break

        if skip:
            continue

        for idx in sorted(remove_indices, reverse=True):
            merged.pop(idx)

        elem = {
            "type": "icon",
            "label": " ".join(absorbed_ocr_labels) if absorbed_ocr_labels else None,
            "bbox_pixel": det["bbox_pixel"],
            "bbox_ratio": icon_ratio,
            "interactable": True,
            "confidence": det["confidence"],
            "source": "yolo+ocr" if absorbed_ocr_labels else "yolo",
        }
        merged.append(elem)

    return merged


# ── Public API ────────────────────────────────────────────────────────

def parse_screen(
    image: Union[Image.Image, np.ndarray],
    *,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    imgsz: int = DEFAULT_IMGSZ,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    use_paddleocr: bool = False,
    caption_icons: bool = True,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> List[ScreenElement]:
    """Parse a GUI screenshot into structured screen elements.

    This is the main entry point for Head 3 (OmniParser-v2 grounding).

    Pipeline:
        1. YOLO detects interactable icons → bounding boxes
        2. OCR detects text regions → bounding boxes + content
        3. Overlap removal merges YOLO + OCR, deduplicating
        4. Florence-2 captions icon crops that lack text labels
        5. Returns a list of ScreenElement objects

    Parameters
    ----------
    image : PIL.Image or np.ndarray
        The screenshot to parse.
    box_threshold : float
        YOLO confidence threshold for icon detection.
    iou_threshold : float
        IoU threshold for overlap removal between detections.
    imgsz : int
        Image size for YOLO detection (longer side, padded).
    text_threshold : float
        OCR confidence threshold.
    use_paddleocr : bool
        Use PaddleOCR instead of EasyOCR.
    caption_icons : bool
        Run Florence-2 on detected icon crops to generate labels.
    max_elements : int
        Maximum number of elements to return.
    cache_dir : str
        Where to cache model weights.

    Returns
    -------
    list[ScreenElement]
        Detected UI elements with bounding boxes and labels.
    """
    t0 = time.time()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    w, h = image.size

    # 1. YOLO icon detection
    yolo_dets = _run_yolo(image, box_threshold=box_threshold, imgsz=imgsz)
    t_yolo = time.time()
    logger.info("YOLO detected %d icons (%.2fs)", len(yolo_dets), t_yolo - t0)

    # 2. OCR text detection
    ocr_texts, ocr_bboxes = _run_ocr(
        image, use_paddleocr=use_paddleocr, text_threshold=text_threshold,
    )
    t_ocr = time.time()
    logger.info("OCR detected %d text regions (%.2fs)", len(ocr_texts), t_ocr - t_yolo)

    # 3. Merge and remove overlaps
    merged = _merge_detections(
        yolo_dets, ocr_texts, ocr_bboxes, (w, h), iou_threshold=iou_threshold,
    )

    # 4. Caption unlabeled icons
    if caption_icons:
        unlabeled = [el for el in merged if el["label"] is None and el["type"] == "icon"]
        if unlabeled:
            icon_ratios = [el["bbox_ratio"] for el in unlabeled]
            captions = _caption_icons(image, icon_ratios)
            for el, cap in zip(unlabeled, captions):
                el["label"] = cap
                el["source"] = "yolo+florence2"
    t_cap = time.time()
    logger.info("Captioned icons (%.2fs), total elements: %d", t_cap - t_ocr, len(merged))

    # 5. Convert to ScreenElement objects, sort by position (top-left first)
    elements = []
    for el in merged:
        bp = el["bbox_pixel"]
        bbox = BBox(x=bp[0], y=bp[1], w=bp[2] - bp[0], h=bp[3] - bp[1])
        label = el.get("label") or "unknown"
        elements.append(ScreenElement(
            element_type=el["type"],
            label=label[:120],
            bbox=bbox,
            interactable=el.get("interactable", el["type"] == "icon"),
            confidence=el.get("confidence", 0.0),
            source=el.get("source", ""),
        ))

    elements.sort(key=lambda e: (e.bbox.y, e.bbox.x))
    elements = elements[:max_elements]

    total = time.time() - t0
    logger.info("parse_screen complete: %d elements in %.2fs", len(elements), total)
    return elements


def parse_screen_annotated(
    image: Union[Image.Image, np.ndarray],
    **kwargs,
) -> Tuple[List[ScreenElement], Image.Image]:
    """Parse screen and return both elements AND an annotated image.

    The annotated image has numbered bounding boxes overlaid, matching
    the element indices in the returned list.

    Returns
    -------
    (elements, annotated_image) : (list[ScreenElement], PIL.Image)
    """
    elements = parse_screen(image, **kwargs)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)

    for i, el in enumerate(elements):
        color = (0, 200, 0) if el.element_type == "text" else (200, 0, 0)
        x1, y1, x2, y2 = el.bbox.to_xyxy()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tag = f"{i}: {el.label[:20]}"
        draw.text((x1 + 2, max(0, y1 - 12)), tag, fill=color)

    return elements, image
