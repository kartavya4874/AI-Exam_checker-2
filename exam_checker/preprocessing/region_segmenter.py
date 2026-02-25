"""
Region segmenter for cropping answer regions from exam pages.

Two strategies:
  A. SAM-based (high-quality scans) — uses facebook/sam-vit-base
  B. Projection profile (fallback) — horizontal ink-gap analysis
"""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

from exam_checker.utils.image_utils import pil_to_numpy, numpy_to_pil, crop_region

logger = logging.getLogger(__name__)

# Type alias for clarity
BBox = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
SegmentResult = Tuple[BBox, Image.Image]


def segment_regions(
    page_image: Image.Image,
    quality_threshold: float = 0.7,
) -> List[SegmentResult]:
    """
    Segment a page into answer regions.

    Chooses strategy based on estimated quality:
      - quality_score > threshold → SAM-based (Strategy A)
      - otherwise → Projection profile (Strategy B)

    Args:
        page_image: PIL Image of the full page.
        quality_threshold: Quality score above which to use SAM.

    Returns:
        List of (bounding_box, cropped_image) tuples sorted in reading order.
    """
    quality_score = _estimate_quality(page_image)
    logger.info("Page quality score: %.2f (threshold: %.2f)", quality_score, quality_threshold)

    if quality_score > quality_threshold:
        try:
            regions = _segment_sam(page_image)
            if regions:
                logger.info("SAM segmentation: %d regions found", len(regions))
                return regions
            logger.warning("SAM returned no regions — falling back to projection")
        except Exception as exc:
            logger.warning("SAM segmentation failed: %s — using projection fallback", exc)

    regions = _segment_projection(page_image)
    logger.info("Projection segmentation: %d regions found", len(regions))
    return regions


def _estimate_quality(image: Image.Image) -> float:
    """
    Estimate scan quality based on DPI-proxy and noise level.

    Args:
        image: PIL Image.

    Returns:
        Quality score between 0.0 and 1.0.
    """
    arr = pil_to_numpy(image)
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr

    h, w = gray.shape

    # Factor 1: Resolution (higher = better)
    resolution_score = min(1.0, (w * h) / (2000 * 3000))

    # Factor 2: Sharpness (Laplacian variance — higher = sharper)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(1.0, laplacian_var / 500.0)

    # Factor 3: Noise estimate (lower = better)
    # Use median-filtered difference
    median = cv2.medianBlur(gray, 5)
    noise = np.mean(np.abs(gray.astype(float) - median.astype(float)))
    noise_score = max(0.0, 1.0 - noise / 30.0)

    quality = (resolution_score * 0.3 + sharpness_score * 0.4 + noise_score * 0.3)
    return float(quality)


def _segment_sam(page_image: Image.Image) -> List[SegmentResult]:
    """
    Segment using SAM (Segment Anything Model) from HuggingFace.

    Args:
        page_image: PIL Image.

    Returns:
        List of (bbox, cropped_image) tuples.
    """
    from transformers import SamModel, SamProcessor
    import torch

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # SAM auto-mask generation using grid-point prompting
    w, h = page_image.size

    # Create a grid of input points
    grid_size = 8
    points = []
    for gy in range(1, grid_size):
        for gx in range(1, grid_size):
            px = int(gx * w / grid_size)
            py = int(gy * h / grid_size)
            points.append([px, py])

    results = []
    seen_regions = []

    for point in points:
        try:
            inputs = processor(
                page_image,
                input_points=[[[point]]],
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = model(**inputs)

            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )

            for mask_tensor in masks:
                for mi in range(mask_tensor.shape[1]):
                    mask = mask_tensor[0, mi].numpy().astype(np.uint8)
                    area = np.sum(mask > 0)
                    total = mask.size

                    # Filter: keep answer-sized regions (1%–50% of page)
                    ratio = area / total
                    if ratio < 0.01 or ratio > 0.50:
                        continue

                    # Get bounding box
                    ys, xs = np.where(mask > 0)
                    if len(xs) == 0:
                        continue
                    bbox = (int(np.min(xs)), int(np.min(ys)),
                            int(np.max(xs)), int(np.max(ys)))

                    # Skip if too similar to existing regions
                    if _is_duplicate_region(bbox, seen_regions, overlap_threshold=0.7):
                        continue

                    seen_regions.append(bbox)
                    cropped = crop_region(page_image, bbox)
                    results.append((bbox, cropped))

        except Exception as exc:
            logger.debug("SAM point %s failed: %s", point, exc)
            continue

    # Sort top-to-bottom, left-to-right
    results.sort(key=lambda r: (r[0][1], r[0][0]))
    return results


def _segment_projection(page_image: Image.Image) -> List[SegmentResult]:
    """
    Segment using horizontal projection profile (gap-based splitting).

    Args:
        page_image: PIL Image.

    Returns:
        List of (bbox, cropped_image) tuples.
    """
    arr = pil_to_numpy(page_image)
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr

    h, w = gray.shape

    # Threshold
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Horizontal projection: sum of ink per row
    h_proj = np.sum(binary, axis=1) / 255.0  # Number of ink pixels per row

    # Normalize
    max_ink = max(np.max(h_proj), 1)
    h_proj_norm = h_proj / max_ink

    # Find gaps (rows with very low ink — < 5% of max)
    gap_threshold = 0.05
    is_gap = h_proj_norm < gap_threshold

    # Find contiguous non-gap blocks
    regions = []
    in_block = False
    block_start = 0
    min_block_height = max(30, h // 50)  # At least 30px or 2% of page

    for row_idx in range(h):
        if not is_gap[row_idx]:
            if not in_block:
                block_start = row_idx
                in_block = True
        else:
            if in_block:
                block_height = row_idx - block_start
                if block_height >= min_block_height:
                    # Find horizontal extent of content in this block
                    block_slice = binary[block_start:row_idx, :]
                    v_proj = np.sum(block_slice, axis=0) / 255.0
                    cols_with_ink = np.where(v_proj > 0)[0]
                    if len(cols_with_ink) > 0:
                        x_min = max(0, int(cols_with_ink[0]) - 10)
                        x_max = min(w, int(cols_with_ink[-1]) + 10)
                    else:
                        x_min, x_max = 0, w

                    bbox = (x_min, block_start, x_max, row_idx)
                    cropped = crop_region(page_image, bbox)
                    regions.append((bbox, cropped))
                in_block = False

    # Handle last block
    if in_block:
        block_height = h - block_start
        if block_height >= min_block_height:
            block_slice = binary[block_start:h, :]
            v_proj = np.sum(block_slice, axis=0) / 255.0
            cols_with_ink = np.where(v_proj > 0)[0]
            if len(cols_with_ink) > 0:
                x_min = max(0, int(cols_with_ink[0]) - 10)
                x_max = min(w, int(cols_with_ink[-1]) + 10)
            else:
                x_min, x_max = 0, w
            bbox = (x_min, block_start, x_max, h)
            cropped = crop_region(page_image, bbox)
            regions.append((bbox, cropped))

    # If no regions found, return the whole page as one region
    if not regions:
        bbox = (0, 0, w, h)
        regions.append((bbox, page_image.copy()))

    return regions


def scan_margins(
    page_image: Image.Image,
    margin_ratio: float = 0.15,
) -> List[SegmentResult]:
    """
    Scan page margins for additional content (answers in margins).

    Checks the leftmost and rightmost portions of the page.

    Args:
        page_image: PIL Image.
        margin_ratio: Fraction of page width to consider as margin.

    Returns:
        List of (bbox, cropped_image) tuples from margins that have content.
    """
    from exam_checker.preprocessing.blank_detector import is_region_blank

    w, h = page_image.size
    margin_px = int(w * margin_ratio)
    results = []

    # Left margin
    left_margin = page_image.crop((0, 0, margin_px, h))
    if not is_region_blank(left_margin, threshold=0.03):
        bbox = (0, 0, margin_px, h)
        results.append((bbox, left_margin))
        logger.info("Content found in left margin")

    # Right margin
    right_margin = page_image.crop((w - margin_px, 0, w, h))
    if not is_region_blank(right_margin, threshold=0.03):
        bbox = (w - margin_px, 0, w, h)
        results.append((bbox, right_margin))
        logger.info("Content found in right margin")

    return results


def _is_duplicate_region(
    bbox: BBox,
    existing: List[BBox],
    overlap_threshold: float = 0.7,
) -> bool:
    """Check if a bbox significantly overlaps with any existing region."""
    for ex in existing:
        overlap = _compute_iou(bbox, ex)
        if overlap > overlap_threshold:
            return True
    return False


def _compute_iou(box1: BBox, box2: BBox) -> float:
    """Compute intersection-over-union of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / max(union, 1)
