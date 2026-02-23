"""
Math analyzer — Pix2Text → LaTeX → SymPy verification pipeline.

Extracts LaTeX from a student's handwritten math, then symbolically
compares it to the answer key using SymPy.  Awards step-by-step partial
credit where possible.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image

from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton Pix2Text instance
# ---------------------------------------------------------------------------
_p2t = None


def _load_p2t():
    """Lazy-load the Pix2Text model."""
    global _p2t
    if _p2t is not None:
        return
    try:
        from pix2text import Pix2Text
        _p2t = Pix2Text()
        log.info("Pix2Text model loaded")
    except Exception as exc:
        log.warning("Pix2Text not available: %s", exc)


# ---------------------------------------------------------------------------
# LaTeX extraction
# ---------------------------------------------------------------------------

def _extract_latex(image: Image.Image) -> str:
    """Run Pix2Text on *image* and return LaTeX string."""
    _load_p2t()
    if _p2t is None:
        return ""
    try:
        result = _p2t.recognize(image)
        # pix2text may return list of dicts or a string depending on version
        if isinstance(result, list):
            parts = []
            for item in result:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return " ".join(parts)
        elif isinstance(result, dict):
            return result.get("text", str(result))
        return str(result)
    except Exception as exc:
        log.warning("Pix2Text recognition failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# SymPy comparison
# ---------------------------------------------------------------------------

def _sympy_compare(student_latex: str, key_latex: str) -> Dict[str, Any]:
    """
    Symbolically compare two LaTeX expressions using SymPy.

    Returns:
        ``{'are_equivalent': bool, 'method': 'sympy'|'fallback', 'error': str|None}``
    """
    from sympy import simplify, S
    from sympy.parsing.latex import parse_latex

    try:
        student_expr = parse_latex(student_latex)
        key_expr = parse_latex(key_latex)
        diff = simplify(student_expr - key_expr)
        are_eq = diff == S.Zero or diff.is_zero
        return {"are_equivalent": bool(are_eq), "method": "sympy", "error": None}
    except Exception as exc:
        log.debug("SymPy comparison failed: %s", exc)
        return {"are_equivalent": False, "method": "fallback", "error": str(exc)}


def _compare_steps(student_latex: str, key_latex: str) -> Dict[str, Any]:
    """
    Compare intermediate steps between two multi-line LaTeX solutions.

    Steps are split by ``\\\\`` or newlines.
    """
    from sympy import simplify, S
    from sympy.parsing.latex import parse_latex

    student_steps = [s.strip() for s in student_latex.replace("\\\\", "\n").split("\n") if s.strip()]
    key_steps = [s.strip() for s in key_latex.replace("\\\\", "\n").split("\n") if s.strip()]

    correct_steps = 0
    total_steps = len(key_steps)

    for i, key_step in enumerate(key_steps):
        if i >= len(student_steps):
            break
        try:
            s_expr = parse_latex(student_steps[i])
            k_expr = parse_latex(key_step)
            diff = simplify(s_expr - k_expr)
            if diff == S.Zero or diff.is_zero:
                correct_steps += 1
        except Exception:
            # If we can't parse, do string comparison
            if student_steps[i].strip() == key_step.strip():
                correct_steps += 1

    return {
        "partial_steps_correct": correct_steps,
        "total_steps": total_steps,
        "student_steps_count": len(student_steps),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_math_answer(
    student_image: Image.Image,
    answer_key_latex: str,
    marks_allocated: int,
) -> Dict[str, Any]:
    """
    Analyse a math answer from a student image.

    Pipeline:
      1. Pix2Text → LaTeX
      2. SymPy symbolic equivalence
      3. Step-by-step partial credit
      4. Compute suggested marks

    Args:
        student_image: Cropped image of the student's math answer.
        answer_key_latex: LaTeX of the correct answer from the key.
        marks_allocated: Total marks for this question.

    Returns:
        Evaluation dict with fields: ``student_latex``, ``key_latex``,
        ``are_equivalent``, ``partial_steps_correct``, ``total_steps``,
        ``suggested_marks``, ``method``, ``feedback``.
    """
    # Step 1 — extract LaTeX
    student_latex = _extract_latex(student_image)
    if not student_latex:
        # Fallback: try TrOCR
        try:
            from exam_checker.ocr.handwriting_ocr import extract_handwritten_text
            trocr_result = extract_handwritten_text(student_image)
            student_latex = trocr_result.get("text", "")
            log.info("Pix2Text failed; using TrOCR text as fallback LaTeX")
        except Exception:
            student_latex = ""

    if not student_latex:
        return {
            "student_latex": "",
            "key_latex": answer_key_latex,
            "are_equivalent": False,
            "partial_steps_correct": 0,
            "total_steps": 1,
            "suggested_marks": 0.0,
            "method": "fallback",
            "feedback": "Could not extract math content from the answer.",
        }

    # Step 2 — symbolic comparison
    comparison = _sympy_compare(student_latex, answer_key_latex)

    # Step 3 — step-by-step partial credit
    steps = _compare_steps(student_latex, answer_key_latex)

    # Step 4 — compute marks
    if comparison["are_equivalent"]:
        suggested_marks = float(marks_allocated)
        feedback = "Answer is mathematically equivalent to the key. Full marks."
    elif steps["total_steps"] > 0:
        step_ratio = steps["partial_steps_correct"] / steps["total_steps"]
        suggested_marks = round(marks_allocated * step_ratio, 1)
        feedback = (
            f"{steps['partial_steps_correct']} of {steps['total_steps']} "
            f"intermediate steps correct. Partial credit awarded."
        )
    else:
        suggested_marks = 0.0
        feedback = "Answer does not match the key and no partial steps could be verified."

    return {
        "student_latex": student_latex,
        "key_latex": answer_key_latex,
        "are_equivalent": comparison["are_equivalent"],
        "partial_steps_correct": steps["partial_steps_correct"],
        "total_steps": steps["total_steps"],
        "suggested_marks": suggested_marks,
        "method": comparison["method"],
        "feedback": feedback,
    }
