"""
Math answer analyzer using Pix2Text and SymPy.

Pipeline: Pix2Text → LaTeX → SymPy symbolic comparison
with partial credit for intermediate steps.
"""

import logging
import re
from typing import Dict, Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_p2t_instance = None


def _get_pix2text():
    """Lazy-load Pix2Text instance."""
    global _p2t_instance
    if _p2t_instance is None:
        try:
            from pix2text import Pix2Text
            _p2t_instance = Pix2Text()
            logger.info("Pix2Text initialized")
        except Exception as exc:
            logger.warning("Pix2Text not available: %s", exc)
    return _p2t_instance


def analyze_math_answer(
    student_image: Image.Image,
    answer_key_latex: str,
    marks_allocated: int,
) -> Dict[str, Any]:
    """
    Analyze a math answer by comparing student work to the answer key.

    Pipeline:
      1. Pix2Text on student image → student LaTeX
      2. SymPy symbolic comparison of final answers
      3. Step-by-step partial credit analysis

    Args:
        student_image: PIL Image of the student's math answer.
        answer_key_latex: LaTeX string of the correct answer.
        marks_allocated: Maximum marks for this question.

    Returns:
        Dict with analysis results including:
          student_latex, key_latex, are_equivalent, partial_steps_correct,
          total_steps, suggested_marks, method, feedback
    """
    result = {
        "student_latex": "",
        "key_latex": answer_key_latex,
        "are_equivalent": False,
        "partial_steps_correct": 0,
        "total_steps": 0,
        "suggested_marks": 0.0,
        "method": "fallback",
        "feedback": "",
        "confidence": 0.0,
    }

    # Step 1: Extract LaTeX from student image
    student_latex = _extract_latex(student_image)
    result["student_latex"] = student_latex

    if not student_latex:
        result["feedback"] = "Could not extract mathematical content from the answer."
        return result

    # Step 2: SymPy symbolic comparison
    equivalence, method = _sympy_compare(student_latex, answer_key_latex)
    result["are_equivalent"] = equivalence
    result["method"] = method

    # Step 3: Step-by-step partial credit
    student_steps = _split_steps(student_latex)
    key_steps = _split_steps(answer_key_latex)
    result["total_steps"] = max(len(key_steps), 1)

    correct_steps = 0
    for i, key_step in enumerate(key_steps):
        if i < len(student_steps):
            step_match, _ = _sympy_compare(student_steps[i], key_step)
            if step_match:
                correct_steps += 1

    result["partial_steps_correct"] = correct_steps

    # Step 4: Calculate marks
    if equivalence:
        result["suggested_marks"] = float(marks_allocated)
        result["feedback"] = "Final answer is correct."
        result["confidence"] = 0.95
    elif correct_steps > 0:
        step_ratio = correct_steps / max(len(key_steps), 1)
        result["suggested_marks"] = round(marks_allocated * step_ratio, 1)
        result["feedback"] = (
            f"{correct_steps}/{len(key_steps)} steps are correct. "
            f"Final answer differs from expected."
        )
        result["confidence"] = 0.7
    else:
        result["suggested_marks"] = 0.0
        result["feedback"] = "No matching steps found. Answer appears incorrect."
        result["confidence"] = 0.6

    return result


def _extract_latex(image: Image.Image) -> str:
    """
    Extract LaTeX from an image using Pix2Text, with TrOCR fallback.

    Args:
        image: PIL Image containing math.

    Returns:
        LaTeX string. Empty string if extraction fails.
    """
    # Try Pix2Text first
    p2t = _get_pix2text()
    if p2t is not None:
        try:
            result = p2t.recognize(image)
            # Pix2Text returns various formats; extract LaTeX
            if isinstance(result, dict):
                latex = result.get("latex", result.get("text", ""))
            elif isinstance(result, str):
                latex = result
            else:
                latex = str(result)
            if latex:
                logger.debug("Pix2Text extracted: %s", latex[:100])
                return latex.strip()
        except Exception as exc:
            logger.warning("Pix2Text failed: %s", exc)

    # Fallback: TrOCR + regex LaTeX reconstruction
    try:
        from exam_checker.ocr.handwriting_ocr import extract_handwritten_text
        text = extract_handwritten_text(image)
        if text:
            latex = _reconstruct_latex(text)
            logger.debug("TrOCR fallback LaTeX: %s", latex[:100] if latex else "empty")
            return latex
    except Exception as exc:
        logger.warning("TrOCR fallback failed: %s", exc)

    return ""


def _reconstruct_latex(text: str) -> str:
    """
    Attempt to reconstruct LaTeX from plain text OCR output.

    Handles common math patterns found in handwritten text.

    Args:
        text: Plain text from OCR.

    Returns:
        Best-effort LaTeX string.
    """
    latex = text

    # Common substitutions
    replacements = [
        (r"(\d+)\s*/\s*(\d+)", r"\\frac{\1}{\2}"),  # fractions
        (r"sqrt\(([^)]+)\)", r"\\sqrt{\1}"),  # sqrt
        (r"\bpi\b", r"\\pi"),
        (r"\btheta\b", r"\\theta"),
        (r"\balpha\b", r"\\alpha"),
        (r"\bbeta\b", r"\\beta"),
        (r"\bgamma\b", r"\\gamma"),
        (r"\bdelta\b", r"\\delta"),
        (r"\bsigma\b", r"\\sigma"),
        (r"\binfinity\b", r"\\infty"),
        (r"\bintegral\b", r"\\int"),
        (r"\bsum\b", r"\\sum"),
        (r"(\w)\^(\d+)", r"\1^{\2}"),  # superscripts
        (r"(\w)_(\d+)", r"\1_{\2}"),  # subscripts
        (r">=", r"\\geq"),
        (r"<=", r"\\leq"),
        (r"!=", r"\\neq"),
        (r"->", r"\\rightarrow"),
    ]

    for pattern, replacement in replacements:
        latex = re.sub(pattern, replacement, latex, flags=re.IGNORECASE)

    return latex


def _sympy_compare(expr1_latex: str, expr2_latex: str) -> tuple:
    """
    Compare two LaTeX expressions using SymPy.

    Args:
        expr1_latex: First LaTeX expression.
        expr2_latex: Second LaTeX expression.

    Returns:
        Tuple of (are_equivalent: bool, method: str).
    """
    try:
        from sympy import simplify, Eq
        from sympy.parsing.latex import parse_latex

        expr1 = parse_latex(expr1_latex)
        expr2 = parse_latex(expr2_latex)

        # Try direct simplification
        diff = simplify(expr1 - expr2)
        if diff == 0:
            return True, "sympy_exact"

        # Try numerical evaluation
        try:
            diff_val = abs(complex(diff.evalf()))
            if diff_val < 1e-10:
                return True, "sympy_numerical"
        except Exception:
            pass

        return False, "sympy_mismatch"

    except Exception as exc:
        logger.debug("SymPy comparison failed: %s", exc)
        # Fall back to string comparison (normalized)
        norm1 = _normalize_latex(expr1_latex)
        norm2 = _normalize_latex(expr2_latex)
        if norm1 and norm2 and norm1 == norm2:
            return True, "string_match"
        return False, "comparison_failed"


def _normalize_latex(latex: str) -> str:
    """Normalize a LaTeX string for string comparison."""
    s = latex.strip()
    s = re.sub(r"\s+", "", s)  # Remove all whitespace
    s = s.replace("{", "").replace("}", "")  # Remove braces
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.lower()
    return s


def _split_steps(latex: str) -> list:
    """
    Split a LaTeX expression into individual steps.

    Splits on \\\\, newlines, or = signs to identify intermediate steps.

    Args:
        latex: LaTeX string potentially containing multiple steps.

    Returns:
        List of individual step LaTeX strings.
    """
    if not latex:
        return []

    # Split on common step delimiters
    steps = re.split(r"\\\\|\n|(?<=[^=!<>])=(?=[^=])", latex)
    steps = [s.strip() for s in steps if s.strip()]
    return steps
