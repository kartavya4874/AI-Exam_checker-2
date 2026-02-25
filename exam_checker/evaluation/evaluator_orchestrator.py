"""
Evaluator orchestrator — routes questions through the full evaluation pipeline.

For each question region:
  1. Classify content type
  2. Run specialized analyzer
  3. Feed results to GPT-4o evaluator
  4. Return final marks + feedback
"""

import logging
from typing import Dict, Any, List, Optional

from PIL import Image

from exam_checker.evaluation.content_classifier import classify_content_type
from exam_checker.evaluation.gpt4o_evaluator import GPT4OEvaluator
from exam_checker.content_analyzers.math_analyzer import analyze_math_answer
from exam_checker.content_analyzers.chemistry_analyzer import (
    analyze_structure as analyze_chem_structure,
    analyze_equation as analyze_chem_equation,
)
from exam_checker.content_analyzers.diagram_analyzer import analyze_diagram
from exam_checker.content_analyzers.code_analyzer import analyze_code
from exam_checker.content_analyzers.text_analyzer import analyze_text
from exam_checker.ocr.ocr_router import route_ocr_with_metadata

logger = logging.getLogger(__name__)


def evaluate_question(
    question_number: str,
    student_image: Image.Image,
    answer_key_data: Dict[str, Any],
    marks_allocated: int,
    openai_client,
    few_shot_examples: Optional[List[Dict]] = None,
    teacher_patterns: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Full evaluation pipeline for a single question.

    Args:
        question_number: e.g. "Q1"
        student_image: PIL Image of student's answer region.
        answer_key_data: Dict with 'text' (and optional 'latex', 'smiles',
            'code', 'image', 'description').
        marks_allocated: Maximum marks for this question.
        openai_client: OpenAI client instance.
        few_shot_examples: GPT-4o few-shot evaluation examples.
        teacher_patterns: Teacher-scored samples for text evaluation.

    Returns:
        Evaluation result dict.
    """
    logger.info("Evaluating %s (marks: %d)", question_number, marks_allocated)

    # Step 1: OCR
    ocr_result = route_ocr_with_metadata(student_image, context="auto")
    student_text = ocr_result["text"]
    ocr_engine = ocr_result["engine"]

    # Step 2: Classify content type
    content_type = classify_content_type(student_image, student_text)
    logger.info("%s classified as: %s (OCR engine: %s)", question_number, content_type, ocr_engine)

    # Step 3: Blank check
    if ocr_result["is_blank"]:
        logger.info("%s is blank/unattempted", question_number)
        return {
            "question_number": question_number,
            "marks_allocated": marks_allocated,
            "marks_obtained": 0,
            "status": "unattempted",
            "content_type": "blank",
            "feedback": "Not Attempted",
            "error_analysis": "",
            "partial_credit_breakdown": {},
            "ocr_engine": "none",
            "pre_analysis": {},
        }

    # Step 4: Run specialized analyzer
    analyzer_results = _run_analyzer(
        content_type, student_image, student_text,
        answer_key_data, marks_allocated, openai_client, teacher_patterns,
    )

    # Step 5: GPT-4o final evaluation
    evaluator = GPT4OEvaluator(openai_client)
    answer_key_text = answer_key_data.get("text", "")

    final_result = evaluator.evaluate_with_context(
        question_number=question_number,
        student_image=student_image,
        student_text=student_text,
        content_type=content_type,
        analyzer_results=analyzer_results,
        answer_key_text=answer_key_text,
        marks_allocated=marks_allocated,
        few_shot_examples=few_shot_examples,
    )

    # Enrich result
    final_result["ocr_engine"] = ocr_engine
    final_result["pre_analysis"] = analyzer_results
    final_result["ocr_text"] = student_text

    logger.info(
        "%s: %s/%d marks (status: %s)",
        question_number,
        final_result.get("marks_obtained", 0),
        marks_allocated,
        final_result.get("status", "unknown"),
    )

    return final_result


def evaluate_student(
    question_images: Dict[str, Image.Image],
    answer_key: Dict[str, Dict[str, Any]],
    marks_map: Dict[str, int],
    openai_client,
    few_shot_examples: Optional[List[Dict]] = None,
    teacher_patterns: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Evaluate all questions for a single student.

    Args:
        question_images: Dict mapping question_number to PIL Image.
        answer_key: Dict mapping question_number to answer data.
        marks_map: Dict mapping question_number to marks allocated.
        openai_client: OpenAI client.
        few_shot_examples: Few-shot patterns.
        teacher_patterns: Teacher samples.

    Returns:
        Dict with per-question results, total marks, and summary.
    """
    results = {}
    total_marks = 0
    total_allocated = 0

    for qnum, image in sorted(question_images.items()):
        key_data = answer_key.get(qnum, {"text": ""})
        marks = marks_map.get(qnum, 10)
        total_allocated += marks

        try:
            result = evaluate_question(
                qnum, image, key_data, marks,
                openai_client, few_shot_examples, teacher_patterns,
            )
            results[qnum] = result
            total_marks += result.get("marks_obtained", 0)
        except Exception as exc:
            logger.error("Evaluation failed for %s: %s", qnum, exc)
            results[qnum] = {
                "question_number": qnum,
                "marks_allocated": marks,
                "marks_obtained": 0,
                "status": "error",
                "content_type": "unknown",
                "feedback": f"Evaluation error: {exc}",
                "error_analysis": str(exc),
                "partial_credit_breakdown": {},
            }

    return {
        "question_results": results,
        "total_marks_obtained": total_marks,
        "total_marks_allocated": total_allocated,
        "percentage": round(total_marks / max(total_allocated, 1) * 100, 1),
    }


def _run_analyzer(
    content_type: str,
    student_image: Image.Image,
    student_text: str,
    answer_key_data: Dict[str, Any],
    marks_allocated: int,
    openai_client,
    teacher_patterns: Optional[List[Dict]],
) -> Dict[str, Any]:
    """Route to the appropriate content analyzer."""
    try:
        if content_type == "math":
            key_latex = answer_key_data.get("latex", answer_key_data.get("text", ""))
            return analyze_math_answer(student_image, key_latex, marks_allocated)

        elif content_type == "chemistry_structure":
            key_smiles = answer_key_data.get("smiles", "")
            return analyze_chem_structure(student_image, key_smiles, marks_allocated)

        elif content_type == "chemistry_equation":
            key_text = answer_key_data.get("text", "")
            return analyze_chem_equation(student_text, key_text, marks_allocated)

        elif content_type == "diagram":
            key_image = answer_key_data.get("image", None)
            key_desc = answer_key_data.get("description", answer_key_data.get("text", ""))
            return analyze_diagram(
                student_image, key_image, key_desc, marks_allocated, openai_client
            )

        elif content_type == "code":
            key_code = answer_key_data.get("code", answer_key_data.get("text", ""))
            test_cases = answer_key_data.get("test_cases", None)
            return analyze_code(
                student_text, key_code, marks_allocated=marks_allocated,
                test_cases=test_cases, openai_client=openai_client,
            )

        else:  # text, mixed, unknown
            key_text = answer_key_data.get("text", "")
            return analyze_text(
                student_text, key_text, teacher_patterns=teacher_patterns,
                marks_allocated=marks_allocated, openai_client=openai_client,
            )

    except Exception as exc:
        logger.error("Analyzer failed for %s: %s", content_type, exc)
        return {"error": str(exc), "method": "analyzer_failed"}
