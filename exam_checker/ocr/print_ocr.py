"""
Tesseract-based printed text OCR.

Wraps pytesseract for extracting printed/typed text from images.
"""

import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def extract_printed_text(
    image: Image.Image,
    lang: str = "eng",
    config: str = "--psm 6",
) -> str:
    """
    Extract printed text from an image using Tesseract OCR.

    Args:
        image: PIL Image containing printed text.
        lang: Language(s) for Tesseract (e.g. "eng", "eng+hin").
        config: Tesseract configuration string.
            --psm 6 = assume a single uniform block of text.

    Returns:
        Extracted text string. Empty string if Tesseract unavailable.
    """
    try:
        import pytesseract
    except ImportError:
        logger.error("pytesseract not installed. Run: pip install pytesseract")
        return ""

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        text = pytesseract.image_to_string(image, lang=lang, config=config)
        text = text.strip()
        logger.debug("Tesseract extracted %d characters", len(text))
        return text

    except Exception as exc:
        logger.error("Tesseract OCR failed: %s", exc)
        return ""


def extract_with_confidence(
    image: Image.Image,
    lang: str = "eng",
) -> dict:
    """
    Extract text with per-word confidence scores using Tesseract.

    Args:
        image: PIL Image.
        lang: Language for Tesseract.

    Returns:
        Dict with:
          - text: full extracted text
          - avg_confidence: average confidence (0-100)
          - word_count: number of detected words
          - low_confidence_words: list of words with confidence < 50
    """
    try:
        import pytesseract
        import pandas as pd
    except ImportError:
        return {
            "text": "",
            "avg_confidence": 0,
            "word_count": 0,
            "low_confidence_words": [],
        }

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DATAFRAME)

        # Filter to actual words (confidence >= 0, non-empty text)
        words = data[(data["conf"] >= 0) & (data["text"].notna())]
        words = words[words["text"].str.strip().astype(bool)]

        if words.empty:
            return {
                "text": "",
                "avg_confidence": 0,
                "word_count": 0,
                "low_confidence_words": [],
            }

        texts = words["text"].tolist()
        confs = words["conf"].tolist()

        full_text = " ".join(str(t) for t in texts)
        avg_conf = sum(confs) / len(confs) if confs else 0
        low_conf = [
            str(t) for t, c in zip(texts, confs)
            if c < 50
        ]

        return {
            "text": full_text,
            "avg_confidence": round(avg_conf, 1),
            "word_count": len(texts),
            "low_confidence_words": low_conf,
        }

    except Exception as exc:
        logger.error("Tesseract detailed OCR failed: %s", exc)
        return {
            "text": extract_printed_text(image, lang),
            "avg_confidence": 0,
            "word_count": 0,
            "low_confidence_words": [],
        }


def is_printed_text(image: Image.Image) -> bool:
    """
    Heuristic check whether an image contains mostly printed (typed) text
    vs. handwritten text.

    Uses pixel regularity analysis: printed text has more uniform spacing
    and sharper edges.

    Args:
        image: PIL Image.

    Returns:
        True if the text appears to be printed.
    """
    import numpy as np

    try:
        gray = image.convert("L")
        arr = np.array(gray)

        # Check edge sharpness via gradient variance
        # Printed text has sharper, more consistent edges
        dy = np.diff(arr.astype(float), axis=0)
        dx = np.diff(arr.astype(float), axis=1)

        gradient_var = np.var(dy) + np.var(dx)

        # Check spacing regularity (printed text has more regular spacing)
        # Project to horizontal axis
        h_proj = np.sum(arr < 128, axis=1)
        nonzero_rows = h_proj[h_proj > 0]

        if len(nonzero_rows) < 5:
            return False

        # Coefficient of variation of line heights
        spacing_cv = np.std(nonzero_rows) / max(np.mean(nonzero_rows), 1)

        # Printed: high gradient variance (sharp edges) + low spacing CV (regular)
        is_print = gradient_var > 1000 and spacing_cv < 0.8

        logger.debug(
            "Printed text check: gradient_var=%.1f, spacing_cv=%.3f → %s",
            gradient_var,
            spacing_cv,
            "printed" if is_print else "handwritten",
        )
        return is_print

    except Exception:
        return False
