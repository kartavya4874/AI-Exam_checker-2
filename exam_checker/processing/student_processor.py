"""
Student processor — handles the full pipeline for a single student.

Pipeline: PDF → images → enhance → segment → OCR + classify + evaluate → results.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from PIL import Image

from exam_checker.ingestion.pdf_converter import convert_pdf_to_images
from exam_checker.preprocessing.scan_enhancer import enhance_pages
from exam_checker.preprocessing.blank_detector import is_page_blank
from exam_checker.preprocessing.region_segmenter import segment_regions
from exam_checker.preprocessing.question_mapper import (
    map_questions_to_regions,
    merge_multi_page_questions,
)
from exam_checker.ocr.ocr_router import route_ocr
from exam_checker.evaluation.evaluator_orchestrator import evaluate_student
from exam_checker.evaluation.grading_engine import calculate_student_results

logger = logging.getLogger(__name__)


def process_student(
    student_file: str,
    course_code: str,
    roll_number: str,
    answer_key: Dict[str, Dict[str, Any]],
    marks_map: Dict[str, int],
    openai_client,
    temp_dir: str = None,
    few_shot_examples: Optional[List[Dict]] = None,
    teacher_patterns: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Full processing pipeline for one student.

    Args:
        student_file: Path to student's answer sheet (PDF/image).
        course_code: Course code string.
        roll_number: Student roll number string.
        answer_key: Dict mapping question_number to answer data.
        marks_map: Dict mapping question_number to marks allocated.
        openai_client: OpenAI client.
        temp_dir: Directory for intermediate files.
        few_shot_examples: GPT-4o few-shot examples.
        teacher_patterns: Teacher evaluation patterns.

    Returns:
        Full result dict with student info, per-question results,
        total marks, grade, and processing status.
    """
    logger.info(
        "Processing student: %s (course: %s, file: %s)",
        roll_number, course_code, student_file,
    )

    result = {
        "course_code": course_code,
        "roll_number": roll_number,
        "file_path": student_file,
        "status": "processing",
        "error": None,
        "pages_processed": 0,
        "questions_found": 0,
    }

    try:
        # Step 1: Convert to images
        pages = convert_pdf_to_images(student_file)
        if not pages:
            result["status"] = "error"
            result["error"] = "No pages extracted from file"
            return result

        result["pages_processed"] = len(pages)
        logger.info("Extracted %d pages", len(pages))

        # Step 2: Enhance pages
        enhanced_pages = enhance_pages(pages, temp_dir=temp_dir)

        # Step 3: Filter blank pages
        non_blank_pages = []
        for i, page in enumerate(enhanced_pages):
            if is_page_blank(page):
                logger.debug("Page %d is blank — skipping", i + 1)
            else:
                non_blank_pages.append(page)

        if not non_blank_pages:
            result["status"] = "blank"
            result["error"] = "All pages appear to be blank"
            result["evaluation"] = calculate_student_results({
                "question_results": {
                    qnum: {
                        "question_number": qnum,
                        "marks_allocated": m,
                        "marks_obtained": 0,
                        "status": "unattempted",
                        "feedback": "Not Attempted",
                    }
                    for qnum, m in marks_map.items()
                },
                "total_marks_obtained": 0,
                "total_marks_allocated": sum(marks_map.values()),
            })
            return result

        # Step 4: Segment regions per page (Parallelized)
        from concurrent.futures import ThreadPoolExecutor

        def process_single_page(page):
            regions = segment_regions(page)
            ocr_text = route_ocr(page, context="auto")
            return map_questions_to_regions(ocr_text, regions, page.height)

        with ThreadPoolExecutor(max_workers=min(len(non_blank_pages), 4)) as executor:
            page_maps = list(executor.map(process_single_page, non_blank_pages))
        
        for i, q_map in enumerate(page_maps):
            logger.debug("Page %d: %d questions", i + 1, len(q_map))

        # Step 5: Merge multi-page questions
        merged_questions = merge_multi_page_questions(page_maps)
        result["questions_found"] = len(merged_questions)
        logger.info("Found %d questions across %d pages", len(merged_questions), len(non_blank_pages))

        # Step 6: Evaluate all questions
        evaluation = evaluate_student(
            question_images=merged_questions,
            answer_key=answer_key,
            marks_map=marks_map,
            openai_client=openai_client,
            few_shot_examples=few_shot_examples,
            teacher_patterns=teacher_patterns,
        )

        # Step 7: Calculate grades
        graded = calculate_student_results(evaluation)
        result["evaluation"] = graded
        result["status"] = "completed"

        logger.info(
            "Student %s: %s/%s marks (%.1f%%, grade: %s)",
            roll_number,
            graded["total_marks_obtained"],
            graded["total_marks_allocated"],
            graded["percentage"],
            graded["grade"],
        )

    except Exception as exc:
        logger.error("Processing failed for %s: %s", roll_number, exc, exc_info=True)
        result["status"] = "error"
        result["error"] = str(exc)

    return result
