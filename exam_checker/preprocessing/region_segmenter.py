"""
Region segmenter — extracts answer regions from a page image.

Two strategies:
  A. SAM-based (facebook/sam-vit-base) for high-quality scans.
  B. Projection-profile based for low-quality scans (fallback).

The strategy is chosen automatically based on estimated image quality.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from exam_checker.utils.image_utils import pil_to_numpy, numpy_to_pil, crop_region
from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

# Type alias
Region = Tuple[Tuple[int, int, int, int], Image.Image]  # (bbox, cropped_image)


# ---------------------------------------------------------------------------
# Quality estimator
# ---------------------------------------------------------------------------

def _estimate_quality(image: Image.Image) -> float:
    """
    Return a 0–1 quality score based on DPI proxy and noise level.

    High DPI + low noise → higher score.
    """
    gray = np.array(image.convert("L"))
    h, w = gray.shape

    # Resolution proxy: higher pixel count → likely higher DPI
    resolution_score = min((h * w) / (2000 * 3000), 1.0)

    # Noise proxy: standard deviation of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_std = laplacian.std()
    # Very high std → noisy. Moderate → good edges.
    noise_score = 1.0 - min(max(lap_std - 30, 0) / 100, 1.0)

    quality = (resolution_score * 0.5 + noise_score * 0.5)
    log.debug("Quality score: %.2f (resolution=%.2f, noise=%.2f)", quality, resolution_score, noise_score)
    return quality


# ---------------------------------------------------------------------------
# Strategy A: SAM-based segmentation
# ---------------------------------------------------------------------------

def _segment_with_sam(image: Image.Image) -> List[Region]:
    """
    Use Facebook SAM (Segment Anything Model) to extract answer regions.

    Loads facebook/sam-vit-base; caches the model on first call.
    Falls back to projection profile on any failure.
    """
    try:
        from transformers import SamModel, SamProcessor
        import torch

        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        model.eval()

        # SAM automatic mask generation via grid-point prompts
        w, h = image.size
        # Create a grid of input points
        grid_points = []
        step_x = max(w // 8, 50)
        step_y = max(h // 8, 50)
        for y in range(step_y, h - step_y, step_y):
            for x in range(step_x, w - step_x, step_x):
                grid_points.append([x, y])

        if not grid_points:
            log.warning("No grid points generated for SAM, falling back")
            return _segment_with_projection(image)

        # Process in batches of points
        regions: List[Region] = []
        seen_bboxes: set = set()
        
        batch_size = 16
        for i in range(0, len(grid_points), batch_size):
            batch_points = grid_points[i:i + batch_size]
            
            for point in batch_points:
                inputs = processor(
                    image,
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
                    for m_idx in range(mask_tensor.shape[1]):
                        mask = mask_tensor[0, m_idx].numpy().astype(np.uint8) * 255
                        
                        # Find bounding box
                        coords = np.where(mask > 0)
                        if len(coords[0]) == 0:
                            continue
                        y_min, y_max = coords[0].min(), coords[0].max()
                        x_min, x_max = coords[1].min(), coords[1].max()
                        
                        area = (y_max - y_min) * (x_max - x_min)
                        page_area = h * w
                        
                        # Filter: keep regions 1%–40% of page
                        if area < page_area * 0.01 or area > page_area * 0.4:
                            continue
                        
                        # Deduplicate (snap to grid)
                        bbox_key = (y_min // 20, x_min // 20, y_max // 20, x_max // 20)
                        if bbox_key in seen_bboxes:
                            continue
                        seen_bboxes.add(bbox_key)
                        
                        bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                        cropped = crop_region(image, bbox)
                        regions.append((bbox, cropped))

        # Sort top-to-bottom, left-to-right
        regions.sort(key=lambda r: (r[0][1], r[0][0]))
        log.info("SAM segmentation: %d regions found", len(regions))
        return regions if regions else _segment_with_projection(image)

    except Exception as exc:
        log.warning("SAM segmentation failed (%s), falling back to projection", exc)
        return _segment_with_projection(image)


# ---------------------------------------------------------------------------
# Strategy B: Projection-profile based
# ---------------------------------------------------------------------------

def _segment_with_projection(image: Image.Image) -> List[Region]:
    """
    Split page into answer regions using horizontal projection profile.

    Algorithm:
      1. Convert to grayscale, threshold.
      2. Compute horizontal projection (sum of ink per row).
      3. Find gaps (runs of near-zero ink rows).
      4. Split image at the gaps.
    """
    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    h, w = binary.shape
    h_proj = np.sum(binary, axis=1)  # sum per row
    max_val = w * 255
    normalised = h_proj / max_val  # 0..1

    # Find gaps: consecutive rows with ink < 1%
    gap_threshold = 0.01
    min_gap_rows = max(10, h // 100)

    in_gap = False
    gap_start = 0
    splits: list[int] = [0]

    for row_idx in range(h):
        is_empty = normalised[row_idx] < gap_threshold
        if is_empty and not in_gap:
            gap_start = row_idx
            in_gap = True
        elif not is_empty and in_gap:
            gap_len = row_idx - gap_start
            if gap_len >= min_gap_rows:
                split_point = gap_start + gap_len // 2
                splits.append(split_point)
            in_gap = False

    splits.append(h)

    regions: List[Region] = []
    for i in range(len(splits) - 1):
        top = splits[i]
        bottom = splits[i + 1]
        if bottom - top < 30:  # too small
            continue
        bbox = (0, top, w, bottom)
        cropped = image.crop(bbox)
        # Skip essentially blank regions
        region_arr = np.array(cropped.convert("L"))
        if np.sum(region_arr < 128) / region_arr.size < 0.005:
            continue
        regions.append((bbox, cropped))

    log.info("Projection segmentation: %d regions found", len(regions))
    return regions


# ---------------------------------------------------------------------------
# Margin scanner (edge case: answers written in margins)
# ---------------------------------------------------------------------------

def _scan_margins(image: Image.Image, margin_pct: float = 0.15) -> List[Region]:
    """Check left and right 15% margins for answer content."""
    w, h = image.size
    margin_px = int(w * margin_pct)
    margin_regions: List[Region] = []

    for label, bbox in [
        ("left", (0, 0, margin_px, h)),
        ("right", (w - margin_px, 0, w, h)),
    ]:
        crop = image.crop(bbox)
        gray = np.array(crop.convert("L"))
        ink_ratio = np.sum(gray < 128) / gray.size
        if ink_ratio > 0.02:
            margin_regions.append((bbox, crop))
            log.info("Found content in %s margin (ink=%.3f)", label, ink_ratio)

    return margin_regions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_page(image: Image.Image) -> List[Region]:
    """
    Segment a page into answer regions using the best available strategy.

    If quality is high enough, uses SAM. Otherwise falls back to
    projection profile. Also scans margins for missed content.

    Args:
        image: Full-page PIL Image (preferably enhanced).

    Returns:
        List of ``((left, top, right, bottom), cropped_image)`` tuples,
        sorted in reading order (top-to-bottom, left-to-right).
    """
    quality = _estimate_quality(image)

    if quality > 0.7:
        regions = _segment_with_sam(image)
    else:
        regions = _segment_with_projection(image)

    # Check margins for missed content
    margin_content = _scan_margins(image)
    if margin_content:
        regions.extend(margin_content)
        # Re-sort
        regions.sort(key=lambda r: (r[0][1], r[0][0]))

    if not regions:
        # Last resort: treat the whole page as one region
        w, h = image.size
        regions = [((0, 0, w, h), image.copy())]
        log.warning("No regions found; treating entire page as one region")

    return regions
