"""
Scan enhancer — OpenCV pipeline for deskewing, denoising, thresholding,
and optional EDSR upscaling of scanned answer sheets.
"""

from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from exam_checker.config import TEMP_DIR
from exam_checker.utils.image_utils import (
    get_image_dpi,
    numpy_to_pil,
    pil_to_numpy,
    save_temp_image,
)
from exam_checker.utils.logger import get_logger

log = get_logger(__name__)


def _compute_skew_angle(gray: np.ndarray) -> float:
    """
    Detect the skew angle of a grayscale image via Canny + HoughLinesP.

    Returns the median angle in degrees. 0.0 if no lines are detected.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return 0.0

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Only keep near-horizontal lines (within ±45°)
        if -45.0 <= angle <= 45.0:
            angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))


def _deskew(image: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Rotate *image* to correct detected skew."""
    angle = _compute_skew_angle(gray)
    if abs(angle) < 0.3:
        log.debug("Skew angle %.2f° too small, skipping deskew", angle)
        return image

    if abs(angle) > 45:
        log.warning("Skew angle %.2f° > 45° — flagging for manual review but attempting correction", angle)

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    log.debug("Deskewed by %.2f°", angle)
    return rotated


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Apply fast non-local means denoising."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Apply Gaussian adaptive thresholding to handle uneven lighting."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=11,
    )


def _upscale_if_needed(image: np.ndarray, dpi: int) -> np.ndarray:
    """
    If DPI < 200, upscale the image.

    First tries EDSR super-resolution if the OpenCV dnn module and weights
    are available.  Falls back to INTER_CUBIC resize on failure.
    """
    if dpi >= 200:
        return image

    scale_factor = max(200 / dpi, 1.5)
    log.info("DPI %d < 200, upscaling by %.1fx", dpi, scale_factor)

    # --- Attempt EDSR ---
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        edsr_path = str(TEMP_DIR / "EDSR_x2.pb")
        sr.readModel(edsr_path)
        sr.setModel("edsr", 2)
        upscaled = sr.upsample(image)
        log.info("Upscaled with EDSR")
        return upscaled
    except Exception:
        log.debug("EDSR unavailable, falling back to INTER_CUBIC resize")

    # --- Fallback: INTER_CUBIC ---
    h, w = image.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _detect_double_sided(gray: np.ndarray) -> bool:
    """
    Heuristic: detect if a page looks like two half-pages scanned together
    (significant horizontal gap near the center).
    """
    h, w = gray.shape
    mid = h // 2
    band = gray[mid - 20: mid + 20, :]
    ink_ratio = np.sum(band < 128) / band.size
    return ink_ratio < 0.005  # almost no ink in the middle band


def enhance_image(image: Image.Image) -> Image.Image:
    """
    Run the full enhancement pipeline on a single page image.

    Steps:
      1. Grayscale conversion
      2. Deskew via Canny + Hough lines
      3. Denoise (fastNlMeansDenoising)
      4. Adaptive threshold
      5. DPI check → upscale if needed
      6. Save enhanced copy for audit

    Args:
        image: Raw PIL Image (one page).

    Returns:
        Enhanced PIL Image.
    """
    bgr = pil_to_numpy(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1 — Deskew (operates on original BGR, uses gray for detection)
    bgr = _deskew(bgr, gray)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 2 — Denoise
    gray = _denoise(gray)

    # 3 — Adaptive threshold
    binary = _adaptive_threshold(gray)

    # 4 — Upscale
    dpi = get_image_dpi(image)
    binary = _upscale_if_needed(binary, dpi)

    # 5 — Convert back to PIL and save audit copy
    enhanced = numpy_to_pil(binary)
    save_temp_image(enhanced, prefix="enhanced_", subdir="enhanced")

    return enhanced


def enhance_pages(images: List[Image.Image]) -> List[Image.Image]:
    """Enhance a list of page images, splitting double-sided pages if detected."""
    enhanced: List[Image.Image] = []
    for idx, img in enumerate(images):
        bgr = pil_to_numpy(img)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        if _detect_double_sided(gray):
            log.info("Page %d appears double-sided — splitting", idx + 1)
            h, w = gray.shape
            top_half = img.crop((0, 0, w, h // 2))
            bottom_half = img.crop((0, h // 2, w, h))
            enhanced.append(enhance_image(top_half))
            enhanced.append(enhance_image(bottom_half))
        else:
            enhanced.append(enhance_image(img))

    log.info("Enhanced %d raw pages → %d output pages", len(images), len(enhanced))
    return enhanced
