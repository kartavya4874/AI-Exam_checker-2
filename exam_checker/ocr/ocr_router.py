"""
OCR router — decides which OCR engine to use for each image region
and returns the best result.

Decision flow:
  1. Blank check → skip if blank.
  2. Pixel regularity analysis → printed vs handwritten.
  3. TrOCR (handwritten) or Tesseract (printed).
  4. EasyOCR for non-Latin character detection.
  5. Return the longest coherent result (more text usually = better OCR).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from exam_checker.preprocessing.blank_detector import is_region_blank
from exam_checker.ocr.handwriting_ocr import extract_handwritten_text
from exam_checker.ocr.print_ocr import extract_printed_text
from exam_checker.ocr.multilingual_ocr import extract_multilingual_text
from exam_checker.utils.logger import get_logger

log = get_logger(__name__)



def _compute_is_printed(image: Image.Image) -> bool:
    """
    Heuristic: printed text has more uniform horizontal projection.
    """
    import cv2
    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    h_proj = np.sum(binary, axis=1).astype(float)

    if h_proj.max() == 0:
        return False

    normalised = h_proj / h_proj.max()
    text_rows = normalised[normalised > 0.05]
    if len(text_rows) < 5:
        return False

    std = float(np.std(text_rows))
    return std < 0.25


def route_ocr(
    image: Image.Image,
    context: str = "handwritten",
) -> str:
    """
    Decide the best OCR engine for *image* and return extracted text.

    Args:
        image: Input PIL Image region.
        context: Hint — ``'handwritten'`` or ``'printed'``.

    Returns:
        Best extracted text string. Empty string if image is blank.
    """
    # 1. Blank check
    if is_region_blank(image):
        log.debug("Region is blank — returning empty text")
        return ""

    results: dict[str, str] = {}

    # 2. Printed vs handwritten
    is_print = _compute_is_printed(image)

    if is_print or context == "printed":
        log.debug("Routing to Tesseract (printed text detected)")
        tess_result = extract_printed_text(image)
        results["tesseract"] = tess_result["text"]
    else:
        log.debug("Routing to TrOCR (handwritten text)")
        trocr_result = extract_handwritten_text(image)
        results["trocr"] = trocr_result["text"]

    # 3. EasyOCR (always run for multilingual detection)
    try:
        easy_result = extract_multilingual_text(image)
        results["easyocr"] = easy_result["text"]

        # If non-Latin detected, bias towards EasyOCR result
        if easy_result["languages_detected"] and any(
            lang not in ("en",) for lang in easy_result["languages_detected"]
        ):
            log.info(
                "Non-Latin characters detected (%s) — merging multilingual result",
                easy_result["languages_detected"],
            )
            # Use EasyOCR as primary for non-Latin
            if len(easy_result["text"]) > len(results.get("trocr", results.get("tesseract", ""))):
                results["primary"] = easy_result["text"]
    except Exception as exc:
        log.warning("EasyOCR failed: %s", exc)

    # 4. Choose best result: longest coherent text wins
    if "primary" in results:
        best = results["primary"]
        engine = "easyocr (primary)"
    else:
        best = ""
        engine = "none"
        for eng, text in results.items():
            if len(text) > len(best):
                best = text
                engine = eng

    log.info("OCR engine used: %s (output length=%d)", engine, len(best))
    return best.strip()
