"""
Multilingual OCR — EasyOCR for non-English scripts (Urdu, Hindi, Arabic).

The EasyOCR reader is initialised once as a singleton for
``['en', 'ur', 'hi', 'ar']``.
"""

from __future__ import annotations

from typing import Dict, List

from PIL import Image
import numpy as np

from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton reader
# ---------------------------------------------------------------------------
_reader = None
_supported_langs = ["en", "ur", "hi", "ar"]


def _load_reader():
    """Lazy-load EasyOCR reader (downloads model weights on first run)."""
    global _reader
    if _reader is not None:
        return

    log.info("Initialising EasyOCR reader for %s…", _supported_langs)
    import easyocr

    _reader = easyocr.Reader(_supported_langs, gpu=False)
    log.info("EasyOCR reader ready")


# ---------------------------------------------------------------------------
# Language detection heuristic
# ---------------------------------------------------------------------------

_DEVANAGARI_RANGE = range(0x0900, 0x097F + 1)
_ARABIC_RANGE = range(0x0600, 0x06FF + 1)
_EXTENDED_ARABIC = range(0x0750, 0x077F + 1)


def _detect_dominant_language(text: str) -> List[str]:
    """Guess dominant scripts from Unicode character ranges."""
    counters = {"latin": 0, "devanagari": 0, "arabic": 0}
    for ch in text:
        cp = ord(ch)
        if ch.isascii() and ch.isalpha():
            counters["latin"] += 1
        elif cp in _DEVANAGARI_RANGE:
            counters["devanagari"] += 1
        elif cp in _ARABIC_RANGE or cp in _EXTENDED_ARABIC:
            counters["arabic"] += 1

    languages: List[str] = []
    if counters["latin"] > 0:
        languages.append("en")
    if counters["devanagari"] > 0:
        languages.append("hi")
    if counters["arabic"] > 0:
        languages.extend(["ur", "ar"])

    return languages if languages else ["en"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_multilingual_text(image: Image.Image) -> Dict[str, object]:
    """
    Extract text from an image supporting multiple scripts.

    Args:
        image: Input PIL Image.

    Returns:
        ``{'text': str, 'languages_detected': list, 'confidence': float}``
    """
    _load_reader()

    img_array = np.array(image.convert("RGB"))
    try:
        results = _reader.readtext(img_array)
    except Exception as exc:
        log.error("EasyOCR readtext failed: %s", exc)
        return {"text": "", "languages_detected": [], "confidence": 0.0}

    if not results:
        return {"text": "", "languages_detected": ["en"], "confidence": 0.0}

    texts: list[str] = []
    confidences: list[float] = []
    for bbox, text, conf in results:
        texts.append(text)
        confidences.append(conf)

    full_text = " ".join(texts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    languages = _detect_dominant_language(full_text)

    log.debug(
        "EasyOCR: %d regions, avg_conf=%.2f, languages=%s",
        len(results),
        avg_conf,
        languages,
    )

    return {
        "text": full_text,
        "languages_detected": languages,
        "confidence": avg_conf,
    }
