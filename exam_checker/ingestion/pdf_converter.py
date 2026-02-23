"""
PDF → per-page PIL Images converter.

Uses PyMuPDF (fitz) as primary engine, with Pillow fallback for
corrupted or password-protected files.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

from PIL import Image

from exam_checker.utils.logger import get_logger

log = get_logger(__name__)


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Convert every page of a PDF to a PIL Image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (default 200).

    Returns:
        List of PIL :class:`~PIL.Image.Image` objects, one per page.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        RuntimeError: If both PyMuPDF and Pillow fail.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.stat().st_size == 0:
        log.error("Zero-byte file: %s", pdf_path)
        return []

    images: List[Image.Image] = []

    # ---- Strategy 1: PyMuPDF ------------------------------------------------
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=matrix)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(img)
            log.debug("Rendered page %d of %s", page_num + 1, pdf_path)

        doc.close()
        log.info("Converted %s → %d pages (PyMuPDF)", pdf_path, len(images))
        return images

    except Exception as exc:
        log.warning("PyMuPDF failed for %s (%s). Trying Pillow fallback…", pdf_path, exc)

    # ---- Strategy 2: Pillow (for simple single-page PDFs or images) ---------
    try:
        img = Image.open(str(path)).convert("RGB")
        images.append(img)
        log.info("Opened %s with Pillow (1 image)", pdf_path)
        return images
    except Exception as exc2:
        log.error("Both PyMuPDF and Pillow failed for %s: %s", pdf_path, exc2)
        raise RuntimeError(
            f"Cannot open file {pdf_path}. It may be corrupted or password-protected."
        ) from exc2


def image_file_to_pil(image_path: str) -> List[Image.Image]:
    """
    Load a single image file (JPG, PNG, TIFF, BMP) as a one-element list
    of PIL Images.  Multi-frame TIFFs produce multiple images.

    Args:
        image_path: Path to the image file.

    Returns:
        List of PIL Images.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if path.stat().st_size == 0:
        log.error("Zero-byte file: %s", image_path)
        return []

    img = Image.open(str(path))
    images: List[Image.Image] = []

    try:
        while True:
            images.append(img.copy().convert("RGB"))
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    log.info("Loaded %s → %d frame(s)", image_path, len(images))
    return images


def load_file_as_images(filepath: str, dpi: int = 200) -> List[Image.Image]:
    """
    Universal loader: PDF → images or image file → images, based on extension.
    """
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return pdf_to_images(filepath, dpi=dpi)
    elif ext in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}:
        return image_file_to_pil(filepath)
    else:
        log.warning("Unsupported file type %s for %s", ext, filepath)
        return []
