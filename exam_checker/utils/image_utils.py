"""
Common image helper functions used across the project.

Handles PIL ↔ NumPy conversions, base64 encoding, resizing, and
temporary file management.
"""

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance

from exam_checker.config import TEMP_DIR
from exam_checker.utils.logger import get_logger

log = get_logger(__name__)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a NumPy array (BGR for OpenCV compatibility)."""
    rgb = np.array(image.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    return bgr


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a NumPy array (BGR or grayscale) to a PIL Image."""
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    rgb = arr[:, :, ::-1]  # BGR → RGB
    return Image.fromarray(rgb, mode="RGB")


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as a base64 string.

    Args:
        image: Source image.
        fmt: Image format (PNG, JPEG, etc.).

    Returns:
        Base64-encoded string (no ``data:`` prefix).
    """
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(b64_str: str) -> Image.Image:
    """Decode a base64 string back to a PIL Image."""
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data))


def resize_image(
    image: Image.Image,
    max_width: int = 1024,
    max_height: int = 1024,
) -> Image.Image:
    """Resize *image* so it fits within *max_width* × *max_height*,
    preserving aspect ratio.  Returns a new image (never mutates the
    original).
    """
    w, h = image.size
    if w <= max_width and h <= max_height:
        return image.copy()
    ratio = min(max_width / w, max_height / h)
    new_size = (int(w * ratio), int(h * ratio))
    return image.resize(new_size, Image.LANCZOS)


def enhance_contrast(image: Image.Image, factor: float = 2.5) -> Image.Image:
    """Increase contrast of *image* by *factor*."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def save_temp_image(
    image: Image.Image,
    prefix: str = "exam_",
    suffix: str = ".png",
    subdir: Optional[str] = None,
) -> str:
    """Save *image* to a temporary file and return the path.

    Args:
        image: PIL Image to save.
        prefix: Filename prefix.
        suffix: File extension.
        subdir: Optional subdirectory under TEMP_DIR.

    Returns:
        Absolute path to the saved file.
    """
    target_dir = TEMP_DIR / subdir if subdir else TEMP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(target_dir))
    os.close(fd)
    image.save(path)
    log.debug("Saved temp image: %s", path)
    return path


def crop_region(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """Crop a region from *image*.

    Args:
        image: Source image.
        bbox: ``(left, upper, right, lower)`` pixel coordinates.

    Returns:
        Cropped PIL Image.
    """
    return image.crop(bbox)


def get_image_dpi(image: Image.Image) -> int:
    """Extract DPI from image metadata; default to 150 if unavailable."""
    try:
        info = image.info
        dpi = info.get("dpi", (150, 150))
        if isinstance(dpi, (list, tuple)):
            return int(dpi[0])
        return int(dpi)
    except Exception:
        return 150


def split_into_strips(
    image: Image.Image,
    strip_height: int = 80,
) -> list[Image.Image]:
    """Split *image* into horizontal strips of *strip_height* pixels.

    Useful for feeding line-sized portions to TrOCR.

    Returns:
        List of PIL Images.
    """
    w, h = image.size
    strips: list[Image.Image] = []
    for y in range(0, h, strip_height):
        bottom = min(y + strip_height, h)
        strip = image.crop((0, y, w, bottom))
        strips.append(strip)
    return strips
