"""
Print OCR — Tesseract wrapper for printed text recognition.

Requires Tesseract to be installed on the system.
"""

from __future__ import annotations

from PIL import Image

from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

_tesseract_available: bool | None = None


def _check_tesseract() -> bool:
    """Verify that Tesseract is installed."""
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available

    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        log.info("Tesseract version: %s", version)
        _tesseract_available = True
    except Exception as exc:
        log.warning("Tesseract not available: %s", exc)
        _tesseract_available = False

    return _tesseract_available


def extract_printed_text(
    image: Image.Image,
    lang: str = "eng",
    psm: int = 6,
) -> dict:
    """
    Extract printed text from an image using Tesseract.

    Args:
        image: Input PIL Image (ideally pre-processed).
        lang: Tesseract language code (default ``eng``).
        psm: Page segmentation mode (6 = uniform block of text).

    Returns:
        ``{'text': str, 'confidence': float, 'engine': 'tesseract'}``
    """
    if not _check_tesseract():
        log.error("Tesseract is not installed — cannot extract printed text")
        return {"text": "", "confidence": 0.0, "engine": "tesseract"}

    import pytesseract

    image = image.convert("RGB")

    custom_config = f"--oem 3 --psm {psm}"
    try:
        # Full text extraction
        text = pytesseract.image_to_string(image, lang=lang, config=custom_config)

        # Confidence from OSD data
        try:
            data = pytesseract.image_to_data(
                image, lang=lang, config=custom_config, output_type=pytesseract.Output.DICT
            )
            confs = [int(c) for c in data["conf"] if int(c) > 0]
            avg_conf = sum(confs) / len(confs) / 100.0 if confs else 0.0
        except Exception:
            avg_conf = 0.5  # default mid-confidence

        log.debug("Tesseract extracted %d chars (confidence=%.2f)", len(text), avg_conf)
        return {"text": text.strip(), "confidence": avg_conf, "engine": "tesseract"}

    except Exception as exc:
        log.error("Tesseract extraction failed: %s", exc)
        return {"text": "", "confidence": 0.0, "engine": "tesseract"}
