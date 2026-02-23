"""
Blank detector — pixel ink-density analysis to detect unattempted
questions without wasting an API call.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from exam_checker.config import BLANK_DETECTION_THRESHOLD
from exam_checker.utils.logger import get_logger

log = get_logger(__name__)


def is_region_blank(
    image_region: Image.Image,
    threshold: float = BLANK_DETECTION_THRESHOLD,
) -> bool:
    """
    Determine whether an image region is blank (unattempted).

    Algorithm:
      1. Convert to grayscale NumPy array.
      2. Binary threshold: pixels < 128 → ink.
      3. ink_ratio = ink_pixels / total_pixels.
      4. If ink_ratio < threshold → blank.
      5. If ink_ratio < 0.005 → blank regardless of threshold.

    Args:
        image_region: Cropped PIL Image of an answer region.
        threshold: Ink density below which the region is blank (default
            from config ``BLANK_DETECTION_THRESHOLD``).

    Returns:
        ``True`` if the region appears blank / unattempted.
    """
    gray = np.array(image_region.convert("L"))
    # ink = dark pixels
    ink_pixels = np.sum(gray < 128)
    total_pixels = gray.size

    if total_pixels == 0:
        return True

    ink_ratio = ink_pixels / total_pixels

    if ink_ratio < 0.005:
        log.debug("Region ink ratio %.4f < 0.005 — blank regardless", ink_ratio)
        return True

    if ink_ratio < threshold:
        log.debug("Region ink ratio %.4f < threshold %.4f — blank", ink_ratio, threshold)
        return True

    log.debug("Region ink ratio %.4f — not blank", ink_ratio)
    return False


def detect_unattempted_questions(
    page_image: Image.Image,
    question_regions: Dict[str, Tuple[int, int, int, int]],
) -> List[str]:
    """
    Identify which questions on a page are unattempted.

    Args:
        page_image: Full-page PIL Image.
        question_regions: Mapping of ``{question_number: (left, upper, right, lower)}``.

    Returns:
        List of question numbers that appear blank.
    """
    blank_questions: List[str] = []

    for q_num, bbox in question_regions.items():
        crop = page_image.crop(bbox)
        if is_region_blank(crop):
            blank_questions.append(q_num)
            log.info("Question %s detected as unattempted (blank)", q_num)

    return blank_questions
