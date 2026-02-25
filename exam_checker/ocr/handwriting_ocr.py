"""
TrOCR-based handwritten text extraction.

Uses microsoft/trocr-large-handwritten from HuggingFace transformers.
Implements singleton pattern for model loading and line-strip splitting
for large images.
"""

import logging
from typing import Optional, Dict
from threading import Lock

from PIL import Image

from exam_checker.utils.image_utils import split_into_strips, enhance_contrast

logger = logging.getLogger(__name__)


class HandwritingOCR:
    """Singleton TrOCR handwritten text extractor."""

    _instance: Optional["HandwritingOCR"] = None
    _lock = Lock()

    def __new__(cls):
        """Ensure only one instance is created."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Load TrOCR processor and model (only once)."""
        if self._initialized:
            return

        logger.info("Loading TrOCR model (microsoft/trocr-large-handwritten)...")
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            self.processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-large-handwritten"
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-large-handwritten"
            )
            self.model.eval()
            self._initialized = True
            logger.info("TrOCR model loaded successfully")
        except Exception as exc:
            logger.error("Failed to load TrOCR: %s", exc)
            self.processor = None
            self.model = None
            self._initialized = True  # Don't retry

    def extract_text(self, image: Image.Image) -> str:
        """
        Extract handwritten text from a PIL Image.

        For large images (>1000px wide), splits into horizontal strips
        and processes each separately.

        Args:
            image: PIL Image containing handwritten text.

        Returns:
            Extracted text string. Empty string if extraction fails.
        """
        if self.model is None or self.processor is None:
            logger.warning("TrOCR not available — returning empty string")
            return ""

        try:
            # Ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            w, h = image.size

            # Large images: split into strips
            if w > 1000 or h > 300:
                return self._extract_from_strips(image)

            # Small images: process directly
            return self._extract_single(image)

        except Exception as exc:
            logger.error("TrOCR extraction failed: %s", exc)
            return ""

    def _extract_single(self, image: Image.Image) -> str:
        """Process a single small image through TrOCR."""
        import torch

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=256)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def _extract_from_strips(self, image: Image.Image) -> str:
        """Split image into strips and process each."""
        strips = split_into_strips(image, strip_height=80, overlap=10)
        lines = []

        for i, strip in enumerate(strips):
            try:
                text = self._extract_single(strip)
                if text:
                    lines.append(text)
            except Exception as exc:
                logger.debug("Strip %d failed: %s", i, exc)
                continue

        return "\n".join(lines)

    def get_confidence(self, image: Image.Image, text: str) -> float:
        """
        Estimate confidence of OCR result.

        If the extracted text is very short relative to the image size,
        confidence is low (may need GPT-4o fallback).

        Args:
            image: Original PIL Image.
            text: Extracted text.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not text:
            return 0.0

        w, h = image.size
        image_area = w * h
        text_length = len(text)

        # Heuristic: expect ~1 char per 200 pixels of image area
        expected_chars = image_area / 200
        if expected_chars == 0:
            return 0.5

        ratio = text_length / expected_chars
        # Ideal ratio is around 0.3-1.0
        if ratio < 0.1:
            return 0.2  # Very little text extracted — likely low quality
        elif ratio > 2.0:
            return 0.6  # Too much text — possible garbage
        else:
            return min(1.0, 0.5 + ratio * 0.5)


# Module-level convenience function
def extract_handwritten_text(image: Image.Image) -> str:
    """
    Extract handwritten text from an image using TrOCR.

    Args:
        image: PIL Image.

    Returns:
        Extracted text string.
    """
    ocr = HandwritingOCR()
    return ocr.extract_text(image)
