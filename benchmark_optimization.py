import time
import logging
from PIL import Image
import numpy as np
from exam_checker.preprocessing.region_segmenter import segment_regions
from exam_checker.ocr.ocr_router import route_ocr
from exam_checker.processing.student_processor import process_student

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_page(width=2000, height=3000):
    """Create a synthetic page with noise to trigger high quality score."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    # Add some high-contrast squares
    for i in range(5):
        x, y = np.random.randint(0, width-500), np.random.randint(0, height-500)
        arr[y:y+500, x:x+500] = 0 # Black box
    return Image.fromarray(arr)

def benchmark_segmentation(num_pages=5):
    logger.info(f"--- Benchmarking Segmentation for {num_pages} pages ---")
    pages = [create_synthetic_page() for _ in range(num_pages)]
    
    start_time = time.time()
    for i, page in enumerate(pages):
        logger.info(f"Segmenting page {i+1}...")
        regions = segment_regions(page, quality_threshold=0.5)
        logger.info(f"Found {len(regions)} regions")
    
    total_time = time.time() - start_time
    avg_time = total_time / num_pages
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average time per page: {avg_time:.2f}s")
    return avg_time

if __name__ == "__main__":
    # Note: This runs with the *currently installed* code (optimized version)
    # To compare, one would have to checkout the previous version.
    # However, since we know it took ~4mins (~240s) per page before, 
    # anything significantly lower confirms the gain.
    
    avg_time = benchmark_segmentation(num_pages=3)
    
    print("\n" + "="*40)
    print("Benchmark Results (Optimized)")
    print("="*40)
    print(f"Average time per page: {avg_time:.2f} seconds")
    print(f"Estimated time for 34 pages: {avg_time * 34 / 60:.2f} minutes")
    print("="*40)
    
    if avg_time < 20:
        print("🎉 SUCCESS: Significant optimization achieved!")
    else:
        print("⚠️  Warning: Optimization could be further improved.")
