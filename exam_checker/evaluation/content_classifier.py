"""
Content type classifier for answer regions.

Classifies regions as: text, math, chemistry_structure, chemistry_equation,
diagram, code, or mixed.
"""

import re
import logging
import numpy as np
import cv2
from PIL import Image

from exam_checker.utils.image_utils import pil_to_numpy

logger = logging.getLogger(__name__)

CODE_KEYWORDS = [
    "def ", "for ", "while ", "int ", "printf", "class ", "return ",
    "import ", "#include", "SELECT ", "function ", "if(", "else{",
    "void ", "public ", "private ", "print(", "console.log",
]

CHEM_ELEMENTS = re.compile(
    r"\b(H|He|Li|Be|B|C|N|O|F|Ne|Na|Mg|Al|Si|P|S|Cl|Ar|K|Ca|Fe|Cu|Zn|Ag|Au|Pb|Hg|Br|I)\b"
)
CHEM_ARROW = re.compile(r"[→⟶]|->|=>")
CHEM_FORMULA = re.compile(r"[A-Z][a-z]?\d*(?:\([A-Z][a-z]?\d*\)\d*)*")

MATH_INDICATORS = [
    r"\\frac", r"\\int", r"\\sum", r"\\sigma", r"\\delta", r"\\alpha",
    r"\\beta", r"\\theta", r"\\pi", r"\\sqrt", r"\\lim", r"\\partial",
]


def classify_content_type(image: Image.Image, text: str) -> str:
    """
    Classify the content type of an answer region.

    Args:
        image: PIL Image of the region.
        text: OCR-extracted text from the region.

    Returns:
        One of: 'text', 'math', 'chemistry_structure', 'chemistry_equation',
        'diagram', 'code', 'mixed'.
    """
    scores = {
        "code": _score_code(text),
        "chemistry_structure": _score_chem_structure(image, text),
        "chemistry_equation": _score_chem_equation(text),
        "math": _score_math(text),
        "diagram": _score_diagram(image, text),
        "text": 0.3,  # Default baseline
    }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Check for mixed content
    high_scores = {k: v for k, v in scores.items() if v > 0.4 and k != best_type}
    if high_scores and best_score < 0.8:
        logger.info("Mixed content: %s", {k: round(v, 2) for k, v in scores.items()})
        return "mixed"

    logger.debug("Classified as '%s' (score=%.2f)", best_type, best_score)
    return best_type


def _score_code(text: str) -> float:
    """Score likelihood of code content."""
    if not text:
        return 0.0
    score = 0.0
    text_lower = text.lower()
    kw_count = sum(1 for kw in CODE_KEYWORDS if kw.lower() in text_lower)
    score += min(kw_count * 0.15, 0.6)

    # Check indentation patterns
    lines = text.split("\n")
    indented = sum(1 for l in lines if l.startswith("    ") or l.startswith("\t"))
    if len(lines) > 3 and indented / len(lines) > 0.3:
        score += 0.3

    return min(score, 1.0)


def _score_chem_structure(image: Image.Image, text: str) -> float:
    """Score likelihood of chemical structure diagram."""
    score = 0.0
    # Check for ring-like shapes via Hough circles
    try:
        arr = pil_to_numpy(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if len(arr.shape) == 3 else arr
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                   param1=100, param2=30, minRadius=10, maxRadius=100)
        if circles is not None:
            score += min(len(circles[0]) * 0.2, 0.5)
    except Exception:
        pass

    # Check for SMILES-like text patterns
    if text and re.search(r"[CNO]\(.*?\)|benzene|ring|organic|structure", text, re.I):
        score += 0.3
    return min(score, 1.0)


def _score_chem_equation(text: str) -> float:
    """Score likelihood of chemical equation."""
    if not text:
        return 0.0
    score = 0.0
    if CHEM_ARROW.search(text):
        score += 0.4
    elements = CHEM_ELEMENTS.findall(text)
    if len(elements) >= 2:
        score += min(len(elements) * 0.1, 0.4)
    if re.search(r"\d+[A-Z][a-z]?\d*\s*\+", text):
        score += 0.2
    return min(score, 1.0)


def _score_math(text: str) -> float:
    """Score likelihood of mathematical content."""
    if not text:
        return 0.0
    score = 0.0
    for indicator in MATH_INDICATORS:
        if re.search(indicator, text, re.I):
            score += 0.15
    if re.search(r"\d+\s*[+\-*/^=]\s*\d+", text):
        score += 0.2
    if re.search(r"[∫∑∏∂√±∞≤≥≠]", text):
        score += 0.3
    return min(score, 1.0)


def _score_diagram(image: Image.Image, text: str) -> float:
    """Score likelihood of diagram (many shapes/lines, little text)."""
    score = 0.0
    try:
        arr = pil_to_numpy(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if len(arr.shape) == 3 else arr
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Diagrams have many edges
        if edge_density > 0.05:
            score += 0.3
        # But little text
        text_len = len(text.strip()) if text else 0
        img_area = image.size[0] * image.size[1]
        text_density = text_len / max(img_area / 1000, 1)
        if text_density < 0.1 and edge_density > 0.03:
            score += 0.4
    except Exception:
        pass
    return min(score, 1.0)
