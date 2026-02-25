"""
Few-shot example builder from teacher-marked sample papers.

Loads teacher samples, extracts their scoring patterns, and formats
them as few-shot examples for GPT-4o prompting.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from PIL import Image

from exam_checker.ocr.ocr_router import route_ocr
from exam_checker.ingestion.pdf_converter import convert_pdf_to_images

logger = logging.getLogger(__name__)


def load_teacher_samples(
    sample_paths: List[str],
    answer_key_text: str,
) -> List[Dict[str, Any]]:
    """
    Load and process teacher-marked sample papers.

    Extracts text and scoring annotations from sample papers to use
    as few-shot evaluation examples.

    Args:
        sample_paths: List of file paths to teacher sample PDFs.
        answer_key_text: Answer key text for context.

    Returns:
        List of few-shot example dicts ready for GPT-4o.
    """
    examples = []

    for path in sample_paths:
        try:
            logger.info("Processing teacher sample: %s", path)
            pages = convert_pdf_to_images(path)

            for page in pages:
                text = route_ocr(page, context="auto")
                if not text.strip():
                    continue

                # Parse annotations (marks, feedback) from the sample
                parsed = _parse_teacher_annotations(text)
                if parsed:
                    example = _format_few_shot(parsed, answer_key_text)
                    examples.append(example)

        except Exception as exc:
            logger.warning("Failed to process sample %s: %s", path, exc)
            continue

    logger.info("Loaded %d few-shot examples from %d samples", len(examples), len(sample_paths))
    return examples


def _parse_teacher_annotations(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse teacher annotations from OCR text of a marked sample.

    Looks for patterns like:
      - Marks: X/Y or X marks
      - Circled/underlined scores
      - Feedback comments

    Args:
        text: OCR text from teacher-marked sample.

    Returns:
        Parsed annotation dict, or None if no annotations detected.
    """
    import re

    annotations = {
        "answer_text": "",
        "marks_awarded": 0,
        "marks_total": 0,
        "feedback_notes": [],
        "question_number": "",
    }

    lines = text.split("\n")
    answer_lines = []
    feedback_lines = []

    for line in lines:
        stripped = line.strip()

        # Detect marks pattern: "5/10", "8 marks", "Score: 7"
        marks_match = re.search(
            r"(\d+)\s*/\s*(\d+)|(\d+)\s*marks?|score\s*:?\s*(\d+)",
            stripped, re.IGNORECASE,
        )
        if marks_match:
            if marks_match.group(1) and marks_match.group(2):
                annotations["marks_awarded"] = int(marks_match.group(1))
                annotations["marks_total"] = int(marks_match.group(2))
            elif marks_match.group(3):
                annotations["marks_awarded"] = int(marks_match.group(3))
            elif marks_match.group(4):
                annotations["marks_awarded"] = int(marks_match.group(4))
            continue

        # Detect question number
        q_match = re.match(r"Q\.?\s*(\d+)|Question\s+(\d+)", stripped, re.IGNORECASE)
        if q_match:
            annotations["question_number"] = q_match.group(1) or q_match.group(2)
            continue

        # Detect teacher feedback (common patterns)
        if any(kw in stripped.lower() for kw in [
            "good", "correct", "wrong", "incorrect", "missing",
            "partial", "well done", "needs", "improve", "error",
        ]):
            feedback_lines.append(stripped)
        else:
            answer_lines.append(stripped)

    annotations["answer_text"] = "\n".join(answer_lines).strip()
    annotations["feedback_notes"] = feedback_lines

    # Only return if we found meaningful annotations
    if annotations["marks_awarded"] > 0 or annotations["feedback_notes"]:
        return annotations
    return None


def _format_few_shot(
    parsed: Dict[str, Any],
    answer_key_text: str,
) -> Dict[str, str]:
    """
    Format parsed annotations into GPT-4o few-shot format.

    Args:
        parsed: Parsed teacher annotations.
        answer_key_text: Answer key text.

    Returns:
        Dict with 'user_content' and 'assistant_content' keys.
    """
    marks_total = parsed["marks_total"] if parsed["marks_total"] > 0 else 10

    user_content = (
        f"Question {parsed['question_number']}:\n"
        f"Answer Key: {answer_key_text[:200]}\n\n"
        f"Student Answer: {parsed['answer_text'][:300]}\n\n"
        f"Marks Allocated: {marks_total}"
    )

    feedback = "; ".join(parsed["feedback_notes"]) if parsed["feedback_notes"] else ""

    assistant_response = json.dumps({
        "question_number": f"Q{parsed['question_number']}",
        "marks_allocated": marks_total,
        "marks_obtained": parsed["marks_awarded"],
        "status": "attempted" if parsed["marks_awarded"] > 0 else "partial",
        "feedback": feedback or f"Awarded {parsed['marks_awarded']}/{marks_total} marks.",
        "partial_credit_breakdown": {},
    })

    return {
        "user_content": user_content,
        "assistant_content": assistant_response,
    }


def build_text_patterns(
    sample_paths: List[str],
) -> List[Dict[str, Any]]:
    """
    Build teacher evaluation patterns for the text analyzer.

    These are used by text_analyzer.py for few-shot prompting.

    Args:
        sample_paths: Paths to teacher sample files.

    Returns:
        List of pattern dicts with question, answer, marks_awarded, feedback.
    """
    patterns = []

    for path in sample_paths:
        try:
            pages = convert_pdf_to_images(path)
            for page in pages:
                text = route_ocr(page, context="auto")
                parsed = _parse_teacher_annotations(text)
                if parsed:
                    patterns.append({
                        "question": f"Q{parsed.get('question_number', '')}",
                        "answer": parsed.get("answer_text", ""),
                        "marks": parsed.get("marks_total", 10),
                        "marks_awarded": parsed.get("marks_awarded", 0),
                        "feedback": "; ".join(parsed.get("feedback_notes", [])),
                    })
        except Exception as exc:
            logger.warning("Failed to build patterns from %s: %s", path, exc)

    return patterns
