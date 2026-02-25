"""
Multilingual OCR using EasyOCR.

Supports English, Urdu, Hindi, Arabic, and other scripts.
Implements singleton pattern for reader initialization.
"""

import logging
from typing import Dict, List, Optional
from threading import Lock

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class MultilingualOCR:
    """Singleton EasyOCR multilingual text extractor."""

    _instance: Optional["MultilingualOCR"] = None
    _lock = Lock()

    SUPPORTED_LANGUAGES = ["en", "ur", "hi", "ar"]

    def __new__(cls):
        """Ensure only one instance is created."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Load EasyOCR reader (only once)."""
        if self._initialized:
            return

        logger.info("Initializing EasyOCR for languages: %s", self.SUPPORTED_LANGUAGES)
        try:
            import easyocr

            self.reader = easyocr.Reader(
                self.SUPPORTED_LANGUAGES,
                gpu=False,  # CPU by default; set True if GPU available
                verbose=False,
            )
            self._initialized = True
            logger.info("EasyOCR initialized successfully")
        except Exception as exc:
            logger.error("Failed to initialize EasyOCR: %s", exc)
            self.reader = None
            self._initialized = True

    def extract_text(self, image: Image.Image) -> Dict:
        """
        Extract multilingual text from a PIL Image.

        Args:
            image: PIL Image containing text.

        Returns:
            Dict with:
              - text: combined extracted text
              - languages_detected: list of detected language codes
              - confidence: average confidence score (0-1)
              - details: list of per-detection dicts with bbox, text, conf
        """
        if self.reader is None:
            logger.warning("EasyOCR not available")
            return {
                "text": "",
                "languages_detected": [],
                "confidence": 0.0,
                "details": [],
            }

        try:
            # Convert to numpy array
            if image.mode != "RGB":
                image = image.convert("RGB")
            arr = np.array(image)

            # Run EasyOCR
            results = self.reader.readtext(arr)

            if not results:
                return {
                    "text": "",
                    "languages_detected": [],
                    "confidence": 0.0,
                    "details": [],
                }

            texts = []
            confidences = []
            details = []

            for bbox, text, conf in results:
                texts.append(text)
                confidences.append(conf)
                details.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": round(conf, 3),
                })

            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences)

            # Detect dominant language from character distribution
            languages = self._detect_languages(combined_text)

            logger.debug(
                "EasyOCR: %d detections, avg_conf=%.2f, langs=%s",
                len(results),
                avg_confidence,
                languages,
            )

            return {
                "text": combined_text,
                "languages_detected": languages,
                "confidence": round(avg_confidence, 3),
                "details": details,
            }

        except Exception as exc:
            logger.error("EasyOCR extraction failed: %s", exc)
            return {
                "text": "",
                "languages_detected": [],
                "confidence": 0.0,
                "details": [],
            }

    @staticmethod
    def _detect_languages(text: str) -> List[str]:
        """
        Detect languages present in text based on character Unicode ranges.

        Args:
            text: Combined extracted text.

        Returns:
            List of detected language codes.
        """
        languages = set()

        for char in text:
            cp = ord(char)
            if 0x0000 <= cp <= 0x007F:
                languages.add("en")  # Basic Latin
            elif 0x0900 <= cp <= 0x097F:
                languages.add("hi")  # Devanagari (Hindi)
            elif 0x0600 <= cp <= 0x06FF:
                languages.add("ar")  # Arabic
            elif 0x0750 <= cp <= 0x077F:
                languages.add("ar")  # Arabic Supplement
            elif 0x0600 <= cp <= 0x06FF or 0xFB50 <= cp <= 0xFDFF:
                languages.add("ur")  # Urdu uses Arabic script + extras

        return sorted(languages) if languages else ["en"]

    def has_non_latin(self, text: str) -> bool:
        """
        Check if text contains non-Latin characters.

        Args:
            text: Text to check.

        Returns:
            True if non-Latin characters are present.
        """
        for char in text:
            cp = ord(char)
            if cp > 0x024F and not char.isspace() and not char.isdigit():
                return True
        return False


# Module-level convenience function
def extract_multilingual_text(image: Image.Image) -> Dict:
    """
    Extract multilingual text from an image using EasyOCR.

    Args:
        image: PIL Image.

    Returns:
        Dict with text, languages_detected, confidence, and details.
    """
    ocr = MultilingualOCR()
    return ocr.extract_text(image)
