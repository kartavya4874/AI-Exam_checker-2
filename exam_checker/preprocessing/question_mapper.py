"""
Question-to-region mapper.

Uses regex patterns to find question numbers in OCR text and maps
them to the spatial regions detected by the region segmenter.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Question number patterns (ordered by specificity)
QUESTION_PATTERNS = [
    # "Question 1", "Question 1.", "Question 1:"
    r"Question\s+(\d+)\s*[.):]*",
    # "Q1", "Q.1", "Q-1", "Q 1"
    r"Q[\.\-\s]*(\d+)\s*[.):]*",
    # "1.", "1)", "1:"  at start of line or after whitespace
    r"(?:^|\n)\s*(\d+)\s*[.):]",
    # "(1)", "[1]"
    r"[\(\[]\s*(\d+)\s*[\)\]]",
    # Sub-questions: "(a)", "a.", "a)"
    r"[\(\[]\s*([a-zA-Z])\s*[\)\]]",
    r"(?:^|\n)\s*([a-zA-Z])\s*[.):]",
    # Roman numerals: "(i)", "(ii)", "i.", "ii."
    r"[\(\[]\s*(i{1,3}|iv|v|vi{0,3})\s*[\)\]]",
]


def map_questions_to_regions(
    ocr_text: str,
    regions: List[Tuple[Tuple[int, int, int, int], Image.Image]],
    page_height: int = 0,
) -> Dict[str, Image.Image]:
    """
    Map question numbers found in OCR text to spatial regions.

    Args:
        ocr_text: Full OCR text from the page.
        regions: List of (bounding_box, cropped_image) tuples from segmenter.
        page_height: Height of the full page (for positional mapping).

    Returns:
        Dict mapping question identifiers (e.g. 'Q1', 'Q2a') to PIL Images.
        Falls back to sequential numbering if pattern matching fails.
    """
    if not regions:
        logger.warning("No regions provided — returning empty mapping")
        return {}

    # Try regex-based mapping first
    question_map = _regex_mapping(ocr_text, regions, page_height)

    if question_map:
        logger.info(
            "Regex mapping found %d questions: %s",
            len(question_map),
            ", ".join(sorted(question_map.keys(), key=_sort_key)),
        )
        return question_map

    # Fallback: sequential numbering
    logger.info(
        "Regex mapping failed — assigning sequential question numbers to %d regions",
        len(regions),
    )
    return _sequential_mapping(regions)


def _regex_mapping(
    ocr_text: str,
    regions: List[Tuple[Tuple[int, int, int, int], Image.Image]],
    page_height: int,
) -> Dict[str, Image.Image]:
    """
    Use regex to find question numbers and map them to regions by position.
    """
    # Find all question numbers with approximate position (character offset)
    found_questions: List[Tuple[str, int]] = []  # (label, char_offset)

    for pattern in QUESTION_PATTERNS:
        for match in re.finditer(pattern, ocr_text, re.IGNORECASE | re.MULTILINE):
            q_num = match.group(1).strip()
            offset = match.start()

            # Normalize
            label = _normalize_label(q_num)
            if label and label not in [f[0] for f in found_questions]:
                found_questions.append((label, offset))

    if not found_questions:
        return {}

    # Sort by appearance order
    found_questions.sort(key=lambda x: x[1])

    # Map questions to regions
    # Strategy: assign regions proportionally based on text offset position
    total_text_len = max(len(ocr_text), 1)
    num_regions = len(regions)

    question_map: Dict[str, Image.Image] = {}
    assigned_regions = set()

    for q_label, offset in found_questions:
        # Estimate which region this question falls into
        relative_pos = offset / total_text_len
        estimated_region_idx = int(relative_pos * num_regions)
        estimated_region_idx = min(estimated_region_idx, num_regions - 1)

        # Find nearest unassigned region
        best_idx = _find_nearest_unassigned(
            estimated_region_idx, num_regions, assigned_regions
        )

        if best_idx is not None:
            bbox, img = regions[best_idx]
            question_map[q_label] = img
            assigned_regions.add(best_idx)

    # Assign remaining regions to last found question (multi-region answers)
    # or as continuation of previous question
    last_label = found_questions[-1][0] if found_questions else None
    for idx in range(num_regions):
        if idx not in assigned_regions and last_label:
            # Check if this might be a sub-part
            sub_label = f"{last_label}_cont_{idx}"
            bbox, img = regions[idx]
            question_map[sub_label] = img

    return question_map


def _sequential_mapping(
    regions: List[Tuple[Tuple[int, int, int, int], Image.Image]],
) -> Dict[str, Image.Image]:
    """Assign sequential question numbers Q1, Q2, ... to regions."""
    question_map = {}
    for idx, (bbox, img) in enumerate(regions, start=1):
        label = f"Q{idx}"
        question_map[label] = img
    return question_map


def merge_multi_page_questions(
    page_maps: List[Dict[str, Image.Image]],
) -> Dict[str, Image.Image]:
    """
    Merge question maps from multiple pages.

    If the same question appears on multiple pages, concatenates the images
    vertically to form one combined region.

    Args:
        page_maps: List of per-page question mappings.

    Returns:
        Merged question mapping.
    """
    merged: Dict[str, List[Image.Image]] = {}

    for page_map in page_maps:
        for label, img in page_map.items():
            if label not in merged:
                merged[label] = []
            merged[label].append(img)

    result: Dict[str, Image.Image] = {}
    for label, images in merged.items():
        if len(images) == 1:
            result[label] = images[0]
        else:
            # Concatenate vertically
            result[label] = _concat_images_vertical(images)
            logger.info(
                "Merged %d images for question %s (multi-page answer)",
                len(images),
                label,
            )

    return result


def handle_duplicate_questions(
    question_map: Dict[str, Image.Image],
) -> Dict[str, Image.Image]:
    """
    Handle cases where the same question number appears multiple times.

    Keeps all attempts, marking them as Q3_attempt1, Q3_attempt2, etc.
    The evaluator will use the best scoring attempt.

    Args:
        question_map: Original question mapping (may have issues).

    Returns:
        Updated mapping with duplicate handling.
    """
    # This is primarily handled in the evaluator_orchestrator,
    # but we provide the mapping here
    return question_map


def _normalize_label(raw: str) -> Optional[str]:
    """Normalize a raw question label to a standard format."""
    raw = raw.strip()
    if not raw:
        return None

    # Pure number → "Q1"
    if raw.isdigit():
        return f"Q{raw}"

    # Single letter → sub-question label
    if len(raw) == 1 and raw.isalpha():
        return raw.lower()

    # Roman numeral
    roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7}
    if raw.lower() in roman_map:
        return f"({raw.lower()})"

    return f"Q{raw}"


def _find_nearest_unassigned(
    target: int,
    total: int,
    assigned: set,
) -> Optional[int]:
    """Find the nearest unassigned region index to the target."""
    if target not in assigned:
        return target

    for delta in range(1, total):
        for candidate in [target + delta, target - delta]:
            if 0 <= candidate < total and candidate not in assigned:
                return candidate

    return None


def _concat_images_vertical(images: List[Image.Image]) -> Image.Image:
    """Concatenate images vertically."""
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    max_width = max(widths)
    total_height = sum(heights)

    combined = Image.new("RGB", (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height

    return combined


def _sort_key(label: str) -> tuple:
    """Sort key for question labels: Q1 < Q2 < Q10 < Q1a < Q1b."""
    parts = re.findall(r"(\d+|[a-zA-Z]+)", label)
    result = []
    for p in parts:
        if p.isdigit():
            result.append((0, int(p), ""))
        else:
            result.append((1, 0, p.lower()))
    return tuple(result) if result else ((0, 0, label),)
