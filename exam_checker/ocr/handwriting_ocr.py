"""
Handwriting OCR — TrOCR pipeline for handwritten text recognition.

Uses ``microsoft/trocr-large-handwritten`` loaded once as a singleton.
Splits large images into line-sized horizontal strips before OCR.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

from PIL import Image

from exam_checker.config import TROCR_MODEL, TROCR_BATCH_SIZE
from exam_checker.utils.image_utils import split_into_strips, enhance_contrast
from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton model holder
# ---------------------------------------------------------------------------
_processor = None
_model = None
_cache: Dict[str, str] = {}  # md5(image_bytes) → extracted text


def _load_model():
    """Lazy-load TrOCR processor and model (only once)."""
    global _processor, _model
    if _processor is not None:
        return

    log.info("Loading TrOCR model: %s (first call, will be cached)…", TROCR_MODEL)
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    _processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    _model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
    _model.eval()
    log.info("TrOCR model loaded successfully")


def _image_hash(image: Image.Image) -> str:
    """Compute a quick hash of image pixel data for caching."""
    data = image.tobytes()
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_handwritten_text(
    image: Image.Image,
    strip_height: int = 80,
    confidence_threshold: float = 0.3,
) -> Dict[str, object]:
    """
    Extract handwritten text from a PIL Image using TrOCR.

    For images wider than 1000 px the image is split into horizontal
    strips of *strip_height* pixels (typical handwritten line height).

    Args:
        image: Input PIL Image.
        strip_height: Height of each line strip in pixels.
        confidence_threshold: Below this ratio of (text length / image area),
            the result is flagged low-confidence.

    Returns:
        ``{'text': str, 'confidence': str, 'low_confidence': bool}``
    """
    _load_model()
    import torch

    # Check cache
    h = _image_hash(image)
    if h in _cache:
        log.debug("Cache hit for image hash %s", h[:8])
        return {"text": _cache[h], "confidence": "cached", "low_confidence": False}

    image = image.convert("RGB")
    w, img_h = image.size

    # Split into strips if large
    if w > 1000 or img_h > 200:
        strips = split_into_strips(image, strip_height)
    else:
        strips = [image]

    lines: list[str] = []
    batch: list[Image.Image] = []

    for strip in strips:
        batch.append(strip)
        if len(batch) >= TROCR_BATCH_SIZE:
            lines.extend(_run_batch(batch))
            batch = []

    if batch:
        lines.extend(_run_batch(batch))

    full_text = "\n".join(line.strip() for line in lines if line.strip())

    # Confidence check
    img_area = w * img_h
    text_len = len(full_text)
    low_conf = (text_len / max(img_area, 1)) < confidence_threshold * 0.001

    if low_conf:
        log.warning(
            "Low confidence OCR result (text_len=%d, area=%d). "
            "May need GPT-4o direct image fallback.",
            text_len,
            img_area,
        )

    _cache[h] = full_text
    return {
        "text": full_text,
        "confidence": "low" if low_conf else "high",
        "low_confidence": low_conf,
    }


def _run_batch(images: list[Image.Image]) -> list[str]:
    """Run TrOCR on a batch of strip images."""
    import torch

    pixel_values = _processor(images=images, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = _model.generate(pixel_values, max_new_tokens=256)
    texts = _processor.batch_decode(generated_ids, skip_special_tokens=True)
    return texts
