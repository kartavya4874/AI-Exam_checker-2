"""
Question mapper — maps question numbers found in OCR text to their
corresponding image regions on the page.

Handles patterns: Q1, Q.1, Q-1, 1., (1), 1), Question 1,
Part (a), (i), i., a.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from PIL import Image

from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

# Type alias — same as region_segmenter
Region = Tuple[Tuple[int, int, int, int], Image.Image]

# ---------------------------------------------------------------------------
# Question-number regex patterns (ordered by specificity)
# ---------------------------------------------------------------------------
_Q_PATTERNS = [
    # Full "Question 3" or "Q 3" or "Q.3" or "Q-3"
    re.compile(r"(?:Question|Ques|Q)\s*[.\-]?\s*(\d+)\s*[.):]\s*", re.IGNORECASE),
    # Sub-questions: (a), (b), (i), (ii)
    re.compile(r"\(([a-zA-Z]|[ivxlcdm]+)\)\s*", re.IGNORECASE),
    # Numbered: 1., 2., 3.  or  1) 2) 3)
    re.compile(r"(?:^|\n)\s*(\d+)\s*[.)]\s+"),
    # Lettered: a. b. c.
    re.compile(r"(?:^|\n)\s*([a-z])\s*[.)]\s+"),
]


def _find_question_labels(text: str) -> List[str]:
    """
    Extract an ordered list of question labels from OCR text.

    Returns labels like ``['Q1', 'Q2', 'Q3a', 'Q3b']``.
    """
    labels: List[str] = []
    current_main: str = ""

    # First pass — main question numbers
    main_pattern = re.compile(
        r"(?:Question|Ques|Q)\s*[.\-]?\s*(\d+)|(?:^|\n)\s*(\d+)\s*[.)]\s+",
        re.IGNORECASE,
    )
    for m in main_pattern.finditer(text):
        num = m.group(1) or m.group(2)
        label = f"Q{num}"
        if label not in labels:
            labels.append(label)
            current_main = num

    # Second pass — sub-questions tied to previous main question
    sub_pattern = re.compile(r"\(([a-zA-Z]|[ivxlcdm]+)\)", re.IGNORECASE)
    for m in sub_pattern.finditer(text):
        sub = m.group(1).lower()
        if current_main:
            label = f"Q{current_main}{sub}"
        else:
            label = f"Q{sub}"
        if label not in labels:
            labels.append(label)

    return labels


def _assign_labels_to_regions(
    labels: List[str],
    regions: List[Region],
) -> Dict[str, Image.Image]:
    """
    Assign discovered question labels to regions in order.

    If there are more labels than regions (line-level detection), merge
    regions.  If more regions than labels, assign sequentially.
    """
    mapping: Dict[str, Image.Image] = {}

    if not labels and not regions:
        return mapping

    if not labels:
        # No questions found → number sequentially
        for i, (bbox, img) in enumerate(regions, start=1):
            mapping[f"Q{i}"] = img
        return mapping

    if len(labels) <= len(regions):
        # One label per region (possibly some regions unlabelled → extras)
        for idx, label in enumerate(labels):
            if idx < len(regions):
                mapping[label] = regions[idx][1]
        # Leftover regions get sequential labels
        for idx in range(len(labels), len(regions)):
            mapping[f"Q{len(labels) + idx - len(labels) + 1}_extra_{idx}"] = regions[idx][1]
    else:
        # More labels than regions — concat regions into first available
        region_idx = 0
        for label in labels:
            if region_idx < len(regions):
                mapping[label] = regions[region_idx][1]
                region_idx += 1
            else:
                # Re-use last region
                mapping[label] = regions[-1][1]

    return mapping


def map_questions_to_regions(
    ocr_text: str,
    regions: List[Region],
) -> Dict[str, Image.Image]:
    """
    Map question numbers extracted from OCR text to spatial image regions.

    Args:
        ocr_text: Full-page OCR text.
        regions: List of ``(bbox, cropped_image)`` from region segmenter.

    Returns:
        ``{'Q1': PIL.Image, 'Q2': PIL.Image, 'Q1a': PIL.Image, …}``
        Falls back to sequential numbering if pattern matching fails.
    """
    labels = _find_question_labels(ocr_text)
    log.debug("Detected question labels: %s", labels)

    if not labels:
        log.warning("No question patterns found in OCR text; assigning sequential labels")

    mapping = _assign_labels_to_regions(labels, regions)
    log.info("Mapped %d questions to %d regions", len(mapping), len(regions))
    return mapping


def handle_multiple_attempts(
    question_map: Dict[str, Image.Image],
) -> Dict[str, Image.Image]:
    """
    If a question number appears more than once (student answered twice),
    keep both. Evaluation should award marks from the better attempt.

    This function checks for duplicate-keyed entries and renames them
    with ``_attempt_1``, ``_attempt_2`` suffixes.
    """
    # Already unique keys from dict, but check for near-duplicates
    # like 'Q3' and 'Q3_extra_3' which both map to Q3
    normalized: Dict[str, List[str]] = {}
    for key in question_map:
        base = re.sub(r"_extra_\d+|_attempt_\d+", "", key)
        normalized.setdefault(base, []).append(key)

    final: Dict[str, Image.Image] = {}
    for base, keys in normalized.items():
        if len(keys) == 1:
            final[base] = question_map[keys[0]]
        else:
            for i, k in enumerate(keys, 1):
                final[f"{base}_attempt_{i}"] = question_map[k]
            log.info("Question %s has %d attempts", base, len(keys))

    return final
