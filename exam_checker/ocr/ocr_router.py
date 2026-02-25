"""
OCR router that decides which OCR engine to use per region.

Routes to the best OCR engine based on content type:
  - Blank → skip
  - Printed text → Tesseract
  - Handwritten → TrOCR
  - Non-Latin → EasyOCR merge
"""

import logging
from typing import Optional

from PIL import Image

from exam_checker.preprocessing.blank_detector import is_region_blank
from exam_checker.ocr.handwriting_ocr import HandwritingOCR, extract_handwritten_text
from exam_checker.ocr.print_ocr import extract_printed_text, is_printed_text
from exam_checker.ocr.multilingual_ocr import MultilingualOCR, extract_multilingual_text

logger = logging.getLogger(__name__)


def route_ocr(
    image: Image.Image,
    context: str = "handwritten",
    blank_threshold: float = 0.02,
) -> str:
    """
    Route an image to the best OCR engine and return extracted text.

    Decision flow:
      1. Blank check → return "" if blank
      2. Is printed text? → Tesseract
      3. Is handwritten? → TrOCR
      4. Run EasyOCR for non-Latin detection
      5. Return best result (longest coherent text)

    Args:
        image: PIL Image of the region to OCR.
        context: Hint about content type ("handwritten", "printed", "auto").
        blank_threshold: Ink ratio threshold for blank detection.

    Returns:
        Extracted text string. Empty string if blank or all OCR fails.
    """
    # Step 1: Blank check
    if is_region_blank(image, threshold=blank_threshold):
        logger.debug("OCR router: region is blank — skipping")
        return ""

    results = {}
    ocr_used = []

    # Step 2: Check if printed text
    if context == "printed" or (context == "auto" and is_printed_text(image)):
        result = extract_printed_text(image)
        if result:
            results["tesseract"] = result
            ocr_used.append("tesseract")
            logger.debug("Tesseract result: %d chars", len(result))

    # Step 3: TrOCR for handwritten
    if context in ("handwritten", "auto") or not results:
        result = extract_handwritten_text(image)
        if result:
            results["trocr"] = result
            ocr_used.append("trocr")
            logger.debug("TrOCR result: %d chars", len(result))

    # Step 4: EasyOCR for multilingual
    multi_result = extract_multilingual_text(image)
    if multi_result["text"]:
        results["easyocr"] = multi_result["text"]
        ocr_used.append("easyocr")
        logger.debug("EasyOCR result: %d chars", len(multi_result["text"]))

        # If non-Latin detected, merge multilingual result
        multilingual_ocr = MultilingualOCR()
        if multilingual_ocr.has_non_latin(multi_result["text"]):
            logger.info("Non-Latin characters detected — using EasyOCR result")
            # Prefer EasyOCR for non-Latin
            best = multi_result["text"]
            logger.info("OCR router used: easyocr (non-Latin)")
            return best

    # Step 5: Select best result
    if not results:
        logger.warning("All OCR engines returned empty results")
        return ""

    # Tiebreaker: longest coherent result
    best_key = max(results, key=lambda k: len(results[k]))
    best = results[best_key]

    logger.info(
        "OCR router result: %s (%d chars) [tried: %s]",
        best_key,
        len(best),
        ", ".join(ocr_used),
    )

    return best


def route_ocr_with_metadata(
    image: Image.Image,
    context: str = "auto",
) -> dict:
    """
    Route OCR and return text with metadata about which engine was used.

    Args:
        image: PIL Image.
        context: Content type hint.

    Returns:
        Dict with:
          - text: extracted text
          - engine: which OCR engine provided the best result
          - all_results: dict of all engine results
          - is_blank: whether the region was blank
          - confidence: estimated confidence
    """
    if is_region_blank(image):
        return {
            "text": "",
            "engine": "none",
            "all_results": {},
            "is_blank": True,
            "confidence": 1.0,
        }

    all_results = {}

    # Tesseract
    try:
        tess = extract_printed_text(image)
        if tess:
            all_results["tesseract"] = tess
    except Exception as exc:
        logger.debug("Tesseract failed: %s", exc)

    # TrOCR
    try:
        trocr = extract_handwritten_text(image)
        if trocr:
            all_results["trocr"] = trocr
    except Exception as exc:
        logger.debug("TrOCR failed: %s", exc)

    # EasyOCR
    try:
        easy = extract_multilingual_text(image)
        if easy["text"]:
            all_results["easyocr"] = easy["text"]
    except Exception as exc:
        logger.debug("EasyOCR failed: %s", exc)

    if not all_results:
        return {
            "text": "",
            "engine": "none",
            "all_results": {},
            "is_blank": False,
            "confidence": 0.0,
        }

    best_key = max(all_results, key=lambda k: len(all_results[k]))
    best_text = all_results[best_key]

    # Confidence heuristic
    ocr = HandwritingOCR()
    confidence = ocr.get_confidence(image, best_text) if best_key == "trocr" else 0.7

    return {
        "text": best_text,
        "engine": best_key,
        "all_results": all_results,
        "is_blank": False,
        "confidence": confidence,
    }
