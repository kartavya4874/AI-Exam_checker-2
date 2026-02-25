"""
Blank detection using pixel ink-density analysis.

Determines whether a region or page is unattempted based on the ratio
of ink pixels to total pixels. No API calls are wasted on blank pages.
"""

import logging
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def is_region_blank(
    image_region: Image.Image,
    threshold: float = 0.02,
) -> bool:
    """
    Determine if an image region is blank (unattempted).

    Uses binary thresholding to detect ink pixels and computes
    the ink-to-total pixel ratio.

    Args:
        image_region: PIL Image of the region to check.
        threshold: Ink ratio below which the region is considered blank.
                   Default 0.02 (2% ink coverage).

    Returns:
        True if the region is blank (unattempted).
    """
    # Convert to grayscale
    if image_region.mode != "L":
        gray = image_region.convert("L")
    else:
        gray = image_region

    arr = np.array(gray)

    if arr.size == 0:
        return True

    # Binary threshold: pixels darker than 128 = ink
    ink_pixels = np.sum(arr < 128)
    total_pixels = arr.size
    ink_ratio = ink_pixels / total_pixels

    # Nearly blank threshold (catches almost-empty pages)
    if ink_ratio < 0.005:
        logger.debug(
            "Region blank (ink_ratio=%.4f < 0.005): definitely empty",
            ink_ratio,
        )
        return True

    if ink_ratio < threshold:
        logger.debug(
            "Region blank (ink_ratio=%.4f < threshold=%.4f)",
            ink_ratio,
            threshold,
        )
        return True

    logger.debug("Region has content (ink_ratio=%.4f)", ink_ratio)
    return False


def is_page_blank(
    page_image: Image.Image,
    threshold: float = 0.01,
) -> bool:
    """
    Determine if an entire page is blank.

    Uses a more lenient threshold for full pages since they include
    margins, headers, etc.

    Args:
        page_image: PIL Image of the full page.
        threshold: Ink ratio threshold (default 0.01 for full pages).

    Returns:
        True if the page is mostly blank.
    """
    return is_region_blank(page_image, threshold=threshold)


def detect_unattempted_questions(
    page_image: Image.Image,
    question_regions: dict,
    threshold: float = 0.02,
) -> List[str]:
    """
    Detect which questions on a page are unattempted (blank).

    Args:
        page_image: PIL Image of the full page.
        question_regions: Dict mapping question_number (str) to
            bounding box tuple (x_min, y_min, x_max, y_max).
        threshold: Ink ratio threshold for blank detection.

    Returns:
        List of question numbers (strings) that are blank/unattempted.
    """
    unattempted = []

    for question_number, bbox in question_regions.items():
        try:
            x_min, y_min, x_max, y_max = bbox
            # Ensure bounds are within image
            w, h = page_image.size
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            if x_max <= x_min or y_max <= y_min:
                logger.warning(
                    "Invalid bbox for %s: %s — marking as unattempted",
                    question_number,
                    bbox,
                )
                unattempted.append(question_number)
                continue

            crop = page_image.crop((x_min, y_min, x_max, y_max))

            if is_region_blank(crop, threshold=threshold):
                unattempted.append(question_number)
                logger.info("Question %s: unattempted (blank)", question_number)
            else:
                logger.debug("Question %s: attempted", question_number)

        except Exception as exc:
            logger.warning(
                "Error checking question %s: %s — skipping",
                question_number,
                exc,
            )

    return unattempted


def compute_ink_density(image: Image.Image) -> float:
    """
    Compute the ink density (ink pixel ratio) of an image.

    Args:
        image: PIL Image.

    Returns:
        Float between 0.0 (all white) and 1.0 (all ink).
    """
    gray = image.convert("L")
    arr = np.array(gray)
    if arr.size == 0:
        return 0.0
    ink_pixels = np.sum(arr < 128)
    return float(ink_pixels / arr.size)


def has_light_handwriting(image: Image.Image) -> bool:
    """
    Check if an image might contain very light handwriting that
    standard thresholds would miss.

    Uses a less aggressive threshold (pixels < 180 instead of 128)
    to detect faint writing.

    Args:
        image: PIL Image.

    Returns:
        True if light handwriting is detected but would be missed
        by standard blank detection.
    """
    gray = image.convert("L")
    arr = np.array(gray)
    if arr.size == 0:
        return False

    # Standard threshold: how much ink at < 128
    standard_ink = np.sum(arr < 128) / arr.size
    # Relaxed threshold: how much ink at < 180
    relaxed_ink = np.sum(arr < 180) / arr.size

    # If standard says blank but relaxed says content, it's light writing
    if standard_ink < 0.02 and relaxed_ink > 0.05:
        logger.info(
            "Light handwriting detected: standard=%.4f, relaxed=%.4f",
            standard_ink,
            relaxed_ink,
        )
        return True

    return False
