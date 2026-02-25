"""
Common image helper utilities.

Provides loading, saving, resizing, base64 encoding, and temporary
file management for PIL Images and numpy arrays.
"""

import io
import os
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


def load_image(path: str) -> Image.Image:
    """
    Load an image from disk as a PIL Image in RGB mode.

    Args:
        path: Path to the image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be opened as an image.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if p.stat().st_size == 0:
        raise IOError(f"Zero-byte file: {path}")
    img = Image.open(str(p))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a numpy array (H, W, C) in BGR for OpenCV.

    Args:
        image: PIL Image.

    Returns:
        numpy array in BGR format.
    """
    rgb = np.array(image)
    if len(rgb.shape) == 2:
        return rgb  # Already grayscale
    return rgb[:, :, ::-1]  # RGB → BGR


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert a numpy array (BGR or grayscale) to a PIL Image in RGB.

    Args:
        arr: numpy array in BGR or grayscale format.

    Returns:
        PIL Image in RGB mode.
    """
    if len(arr.shape) == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr[:, :, ::-1], mode="RGB")  # BGR → RGB


def resize_image(
    image: Image.Image,
    max_width: int = 2000,
    max_height: int = 2000,
) -> Image.Image:
    """
    Resize an image if it exceeds max dimensions, preserving aspect ratio.

    Args:
        image: PIL Image.
        max_width: Maximum width in pixels.
        max_height: Maximum height in pixels.

    Returns:
        Resized PIL Image (or original if already within limits).
    """
    w, h = image.size
    if w <= max_width and h <= max_height:
        return image

    ratio = min(max_width / w, max_height / h)
    new_size = (int(w * ratio), int(h * ratio))
    return image.resize(new_size, Image.LANCZOS)


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """
    Encode a PIL Image as a base64 string.

    Args:
        image: PIL Image.
        fmt: Image format (PNG, JPEG, etc.).

    Returns:
        Base64-encoded string.
    """
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """
    Decode a base64 string into a PIL Image.

    Args:
        b64_string: Base64-encoded image data.

    Returns:
        PIL Image.
    """
    data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(data)).convert("RGB")


def save_temp_image(
    image: Image.Image,
    temp_dir: str = None,
    prefix: str = "exam_",
    suffix: str = ".png",
) -> str:
    """
    Save a PIL Image to a temporary file and return the path.

    Args:
        image: PIL Image.
        temp_dir: Directory for temp files. Defaults to system temp.
        prefix: Filename prefix.
        suffix: Filename extension.

    Returns:
        Absolute path to the saved temporary file.
    """
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=temp_dir)
    os.close(fd)
    image.save(path)
    logger.debug("Saved temp image: %s", path)
    return path


def crop_region(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Crop a region from a PIL Image.

    Args:
        image: PIL Image.
        bbox: Bounding box as (x_min, y_min, x_max, y_max).

    Returns:
        Cropped PIL Image.
    """
    x1, y1, x2, y2 = bbox
    w, h = image.size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return image.crop((x1, y1, x2, y2))


def enhance_contrast(
    image: Image.Image,
    factor: float = 2.5,
) -> Image.Image:
    """
    Enhance image contrast.

    Args:
        image: PIL Image.
        factor: Contrast enhancement factor (1.0 = original).

    Returns:
        Contrast-enhanced PIL Image.
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def get_image_dpi(image: Image.Image) -> Tuple[int, int]:
    """
    Extract DPI from image metadata.

    Args:
        image: PIL Image.

    Returns:
        Tuple of (x_dpi, y_dpi). Defaults to (150, 150) if not found.
    """
    try:
        dpi = image.info.get("dpi", (150, 150))
        if isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
            return (int(dpi[0]), int(dpi[1]))
        return (150, 150)
    except Exception:
        return (150, 150)


def split_into_strips(
    image: Image.Image,
    strip_height: int = 80,
    overlap: int = 10,
) -> list:
    """
    Split an image into horizontal strips for line-by-line OCR.

    Args:
        image: PIL Image.
        strip_height: Height of each strip in pixels.
        overlap: Overlap between consecutive strips in pixels.

    Returns:
        List of PIL Image strips.
    """
    w, h = image.size
    strips = []
    y = 0
    while y < h:
        bottom = min(y + strip_height, h)
        strip = image.crop((0, y, w, bottom))
        if bottom - y > 10:  # Skip very thin strips
            strips.append(strip)
        y += strip_height - overlap
    return strips
