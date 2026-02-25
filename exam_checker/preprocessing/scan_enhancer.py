"""
Scan enhancement pipeline using OpenCV.

Full pipeline: grayscale → deskew → denoise → adaptive threshold → DPI upscale.
Handles rotated scans, low-DPI images, and uneven lighting.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from exam_checker.utils.image_utils import (
    load_image,
    pil_to_numpy,
    numpy_to_pil,
    get_image_dpi,
    save_temp_image,
)

logger = logging.getLogger(__name__)


def enhance_image(
    image: Image.Image,
    temp_dir: str = None,
    save_audit: bool = True,
) -> Image.Image:
    """
    Full scan enhancement pipeline.

    Steps:
        1. Convert to grayscale
        2. Deskew (correct rotation)
        3. Denoise
        4. Adaptive threshold
        5. DPI upscale if needed
        6. Optionally save enhanced image for audit

    Args:
        image: Input PIL Image.
        temp_dir: Directory for saving audit copies.
        save_audit: Whether to save the enhanced image.

    Returns:
        Enhanced PIL Image.
    """
    # Convert PIL → numpy (BGR)
    img_bgr = pil_to_numpy(image)
    h, w = img_bgr.shape[:2]
    logger.debug("Enhancement input: %dx%d", w, h)

    # Step 1: Grayscale
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    # Step 2: Deskew
    gray, rotation_angle = _deskew(gray)
    if abs(rotation_angle) > 0.1:
        logger.info("Deskewed by %.2f°", rotation_angle)

    # Step 3: Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 4: Adaptive threshold
    enhanced = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=11,
    )

    # Step 5: DPI upscale
    dpi_x, dpi_y = get_image_dpi(image)
    if dpi_x < 200 or dpi_y < 200:
        scale = max(200 / dpi_x, 200 / dpi_y)
        enhanced = _upscale(enhanced, scale)
        logger.info("Upscaled from %d DPI (scale=%.2f)", min(dpi_x, dpi_y), scale)

    # Convert back to PIL
    result = numpy_to_pil(enhanced)

    # Step 6: Save audit copy
    if save_audit and temp_dir:
        save_temp_image(result, temp_dir=temp_dir, prefix="enhanced_")

    return result


def enhance_file(
    file_path: str,
    temp_dir: str = None,
) -> Image.Image:
    """
    Enhance an image file from disk.

    Args:
        file_path: Path to the image.
        temp_dir: Directory for audit copies.

    Returns:
        Enhanced PIL Image.
    """
    image = load_image(file_path)
    return enhance_image(image, temp_dir=temp_dir)


def enhance_pages(
    pages: List[Image.Image],
    temp_dir: str = None,
) -> List[Image.Image]:
    """
    Enhance a list of page images.

    Args:
        pages: List of PIL Images.
        temp_dir: Directory for audit copies.

    Returns:
        List of enhanced PIL Images.
    """
    enhanced = []
    for i, page in enumerate(pages):
        try:
            result = enhance_image(page, temp_dir=temp_dir, save_audit=False)
            enhanced.append(result)
            logger.debug("Enhanced page %d/%d", i + 1, len(pages))
        except Exception as exc:
            logger.warning("Enhancement failed for page %d: %s — using original", i + 1, exc)
            enhanced.append(page)
    return enhanced


def _deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Deskew a grayscale image using Hough line detection.

    Detects edges, finds lines, computes median angle, and rotates.

    Args:
        image: Grayscale numpy array.

    Returns:
        Tuple of (deskewed_image, rotation_angle_degrees).
    """
    h, w = image.shape[:2]

    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Find lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=w // 4,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        logger.debug("No lines detected — skipping deskew")
        return image, 0.0

    # Compute angles
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > 0:
            angle = np.degrees(np.arctan2(dy, dx))
            # Only consider nearly-horizontal lines (±30°)
            if abs(angle) < 30:
                angles.append(angle)

    if not angles:
        logger.debug("No suitable angles found — skipping deskew")
        return image, 0.0

    median_angle = float(np.median(angles))

    # Flag extreme rotations
    if abs(median_angle) > 45:
        logger.warning(
            "Rotation > 45° detected (%.1f°) — flagged for manual review",
            median_angle,
        )

    # Rotate
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated, median_angle


def _upscale(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Upscale an image using EDSR if available, otherwise INTER_CUBIC.

    Args:
        image: Grayscale or BGR numpy array.
        scale: Scale factor (e.g. 2.0 for doubling).

    Returns:
        Upscaled numpy array.
    """
    # Try EDSR super-resolution (optional)
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "model_cache", "EDSR_x2.pb"
        )
        if os.path.exists(model_path):
            sr.readModel(model_path)
            sr.setModel("edsr", 2)
            # EDSR requires 3-channel input
            if len(image.shape) == 2:
                img_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                upscaled = sr.upsample(img_3ch)
                return cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            return sr.upsample(image)
    except Exception as exc:
        logger.debug("EDSR not available, using INTER_CUBIC: %s", exc)

    # Fallback: cv2.resize
    new_w = int(image.shape[1] * scale)
    new_h = int(image.shape[0] * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def detect_double_sided(image: Image.Image) -> bool:
    """
    Detect if a scan appears to be double-sided (two pages on one scan).

    Looks for a vertical gap or fold line in the middle of the image.

    Args:
        image: PIL Image.

    Returns:
        True if the scan appears to contain two half-pages.
    """
    arr = pil_to_numpy(image)
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr

    h, w = gray.shape
    # Check vertical strip in the middle 10% of the image
    mid_start = int(w * 0.45)
    mid_end = int(w * 0.55)
    mid_strip = gray[:, mid_start:mid_end]

    # If middle strip is mostly white (> 80% white pixels), it's likely a fold
    white_ratio = np.sum(mid_strip > 200) / mid_strip.size
    return white_ratio > 0.80


def split_double_sided(image: Image.Image) -> List[Image.Image]:
    """
    Split a double-sided scan into two separate page images.

    Args:
        image: PIL Image that contains two half-pages.

    Returns:
        List of two PIL Images (left half, right half).
    """
    w, h = image.size
    mid = w // 2
    left = image.crop((0, 0, mid, h))
    right = image.crop((mid, 0, w, h))
    return [left, right]


def detect_crossed_out_regions(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect crossed-out (strikethrough) regions in an image.

    Looks for dense diagonal or horizontal lines that suggest
    the student has crossed out an answer.

    Args:
        image: Grayscale numpy array.

    Returns:
        List of bounding boxes (x, y, w, h) of crossed-out regions.
    """
    h, w = image.shape[:2]
    crossed = []

    # Detect lines using HoughLinesP
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=50, minLineLength=w // 6, maxLineGap=10,
    )

    if lines is None:
        return crossed

    # Group diagonal/crossing lines
    cross_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # Diagonal lines (20°–70°) indicate crossing out
        if 20 < angle < 70:
            cross_lines.append((x1, y1, x2, y2))

    # Cluster nearby diagonal lines into regions
    if len(cross_lines) >= 2:
        all_points = []
        for x1, y1, x2, y2 in cross_lines:
            all_points.extend([(x1, y1), (x2, y2)])
        points = np.array(all_points)

        x_min = int(np.min(points[:, 0]))
        y_min = int(np.min(points[:, 1]))
        x_max = int(np.max(points[:, 0]))
        y_max = int(np.max(points[:, 1]))

        region_w = x_max - x_min
        region_h = y_max - y_min

        # Only flag if region is significant size
        if region_w > w * 0.1 and region_h > h * 0.05:
            crossed.append((x_min, y_min, region_w, region_h))

    return crossed
