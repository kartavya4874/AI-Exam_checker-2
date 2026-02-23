"""
Content classifier — Classifies answer regions into specialized types.

Returns one of: 'text', 'math', 'chemistry_structure', 
                'chemistry_equation', 'diagram', 'code', 'mixed'

Classification rules use a mix of OCR-text keyword detection and 
OpenCV-based image feature analysis.
"""

import re
from typing import Dict, List, Optional
import numpy as np
import cv2
from PIL import Image

from exam_checker.utils.logger import get_logger
from exam_checker.utils.image_utils import pil_to_numpy

log = get_logger(__name__)

def classify_content_type(image: Image.Image, text: str) -> str:
    """
    Classify the content type of an answer region.

    Args:
        image: PIL Image of the answer region.
        text: Extracted OCR text from the region.

    Returns:
        One of: 'text', 'math', 'chemistry_structure', 
                'chemistry_equation', 'diagram', 'code', 'mixed'
    """
    text = text.lower()
    
    # 1. CODE Detection
    # Keywords and common syntax patterns
    code_keywords = [
        'def', 'for', 'while', 'int', 'printf', 'class', 'return', 
        'import', '#include', 'select', 'function', 'const', 'let', 'var'
    ]
    code_score = sum(1 for kw in code_keywords if re.search(rf'\b{kw}\b', text))
    if code_score >= 2 or (code_score >= 1 and ('{' in text or '(' in text or '=' in text)):
        log.debug("Classified as 'code' based on keyword score: %d", code_score)
        return 'code'

    # 2. CHEMISTRY DETECTION (Structure vs Equation)
    # Check for equation-like patterns: arrows, element symbols with numbers
    chem_eq_pattern = r'[A-Z][a-z]?\d*.*[→⟶=].*[A-Z][a-z]?\d*'
    if re.search(chem_eq_pattern, text) or '→' in text or '⟶' in text:
        log.debug("Classified as 'chemistry_equation' based on arrow/pattern")
        return 'chemistry_equation'

    # Check for ring-like shapes or SMILES-like patterns for structures
    if re.search(r'[C|N|O|H]\d*\(', text) or any(w in text for w in ['benzene', 'ring', 'bond', 'structure']):
        log.debug("Classified as 'chemistry_structure' based on keywords/SMILES")
        return 'chemistry_structure'

    # Image analysis for rings using Hough Circles
    # Convert to grayscale numpy for CV2
    bgr = pil_to_numpy(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, 1, 20,
        param1=50, param2=30, minRadius=10, maxRadius=50
    )
    if circles is not None:
        log.debug("Classified as 'chemistry_structure' based on detected circles (rings)")
        return 'chemistry_structure'

    # 3. MATH Detection
    # LaTeX symbols, math operators, or numbers with superscripts/subscripts
    math_symbols = ['\\frac', '\\int', '\\sigma', '\\delta', '^', '_', '=', '+', '-', '*', '/']
    math_score = sum(1 for sym in math_symbols if sym in text)
    if math_score >= 2 or re.search(r'\d+\^|x\^|y\^', text):
        log.debug("Classified as 'math' based on symbol score: %d", math_score)
        return 'math'

    # 4. DIAGRAM Detection
    # High ink density but low text density (ink present but OCR returns little text)
    gray_arr = np.array(image.convert('L'))
    ink_pixels = np.sum(gray_arr < 128)
    total_pixels = gray_arr.size
    ink_density = ink_pixels / total_pixels
    
    text_density = len(text.strip()) / max(total_pixels, 1) * 1000 # chars per 1k pixels
    
    if ink_density > 0.05 and text_density < 0.5:
        log.debug("Classified as 'diagram' based on ink density (%.2f) and low text density (%.2f)", ink_density, text_density)
        return 'diagram'

    # 5. MIXED / FALLBACK
    # If some text exists but multiple types could apply
    if (code_score + math_score) >= 3:
        return 'mixed'

    log.debug("Falling back to 'text' classification")
    return 'text'
