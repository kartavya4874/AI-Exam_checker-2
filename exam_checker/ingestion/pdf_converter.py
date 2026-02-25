"""
PDF to per-page PIL Image converter.

Handles normal PDFs, corrupt PDFs (graceful fallback), password-protected
PDFs (catches and logs), and image files (pass-through).
"""

import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


def convert_pdf_to_images(
    file_path: str,
    dpi: int = 200,
) -> List[Image.Image]:
    """
    Convert a PDF or image file to a list of PIL Images (one per page).

    Args:
        file_path: Path to the PDF or image file.
        dpi: Resolution for PDF rendering (default 200).

    Returns:
        List of PIL Images, one per page.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be opened after all fallback attempts.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.stat().st_size == 0:
        raise IOError(f"Zero-byte file: {file_path}")

    ext = path.suffix.lower()

    # ── Image files: direct load ──
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
        return _load_image_file(file_path)

    # ── PDF files: use PyMuPDF ──
    if ext == ".pdf":
        return _convert_pdf_pymupdf(file_path, dpi)

    # ── Unknown format ──
    logger.warning("Unsupported file format: %s — attempting as image", ext)
    return _load_image_file(file_path)


def _load_image_file(file_path: str) -> List[Image.Image]:
    """
    Load a single image file as a list with one PIL Image.

    Args:
        file_path: Path to the image file.

    Returns:
        List containing one PIL Image in RGB mode.
    """
    try:
        img = Image.open(file_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        logger.info("Loaded image: %s (%dx%d)", file_path, img.width, img.height)
        return [img]
    except Exception as exc:
        logger.error("Failed to load image %s: %s", file_path, exc)
        raise IOError(f"Cannot open image: {file_path}") from exc


def _convert_pdf_pymupdf(
    file_path: str,
    dpi: int = 200,
) -> List[Image.Image]:
    """
    Convert a PDF to per-page PIL Images using PyMuPDF (fitz).

    Handles password-protected and corrupt PDFs gracefully.

    Args:
        file_path: Path to the PDF file.
        dpi: Rendering resolution.

    Returns:
        List of PIL Images, one per page.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) not installed. Run: pip install PyMuPDF")
        raise

    images: List[Image.Image] = []

    try:
        doc = fitz.open(file_path)
    except fitz.FileDataError as exc:
        # Password-protected or severely corrupt
        logger.error(
            "Cannot open PDF (may be password-protected or corrupt): %s — %s",
            file_path,
            exc,
        )
        raise IOError(
            f"PDF cannot be opened (password-protected or corrupt): {file_path}"
        ) from exc
    except Exception as exc:
        logger.error("Error opening PDF %s: %s", file_path, exc)
        # Fallback: try opening as image (some files have wrong extension)
        try:
            return _load_image_file(file_path)
        except Exception:
            raise IOError(f"Cannot open file: {file_path}") from exc

    try:
        zoom = dpi / 72.0  # PyMuPDF default is 72 DPI
        mat = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=mat)

                # Convert pixmap to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                logger.debug(
                    "PDF page %d/%d: %dx%d",
                    page_num + 1,
                    len(doc),
                    pix.width,
                    pix.height,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to render page %d of %s: %s — skipping",
                    page_num + 1,
                    file_path,
                    exc,
                )
                continue

    finally:
        doc.close()

    if not images:
        raise IOError(f"No pages could be rendered from PDF: {file_path}")

    logger.info(
        "Converted PDF: %s → %d pages at %d DPI",
        file_path,
        len(images),
        dpi,
    )
    return images


def get_page_count(file_path: str) -> int:
    """
    Get the number of pages in a PDF (or 1 for image files).

    Args:
        file_path: Path to the file.

    Returns:
        Number of pages.
    """
    ext = Path(file_path).suffix.lower()
    if ext != ".pdf":
        return 1

    try:
        import fitz

        doc = fitz.open(file_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 1
